"""
Sequential tool-call ReAct loop for LLaMA 3 models.

LLaMA 3's chat template explicitly rejects multiple parallel tool calls.
This module provides a patched version of the ReAct loop that processes
ALL tool calls, but sequentially - each tool call becomes its own
conversation turn with a separate AIMessage.

Background:
- LLaMA 3 8B's Jinja chat template raises: "This model only supports single tool-calls at once!"
- This happens when an AIMessage contains multiple tool_calls
- The fix: split multiple tool calls into separate AIMessage/ToolMessage pairs

References:
- https://github.com/vllm-project/vllm/pull/12938
- https://docs.vllm.ai/en/latest/features/tool_calling/

Usage:
    Replace calls to `_run_react_json_agent` with `run_react_sequential_tools`:

    from Agents.react_single_tool import run_react_sequential_tools

    result = run_react_sequential_tools(
        llm=llm,
        system_prompt=AGENT2_SYSTEM,
        user_payload=payload,
        tools=[get_matching_content],
        max_rounds=12,
    )
"""

import json
import logging
import re
from typing import Any, Dict, List

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    BaseMessage,
)

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> dict:
    """Parse a JSON object from an LLM response (defensive)."""
    if text is None:
        raise ValueError("Empty response from model")
    raw = text.strip()

    # Strip ``` fences if present
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json", "", 1).strip()

    # Try direct load
    try:
        return json.loads(raw)
    except Exception:
        pass

    # Try to locate first {...} block
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        raise ValueError(f"Model did not return JSON. Raw output:\n{raw}")
    # Use raw_decode to handle extra trailing data from the model
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(m.group(0))
    return obj


def run_react_sequential_tools(
    *,
    llm,
    system_prompt: str,
    user_payload: Dict[str, Any],
    tools: List[Any],
    max_rounds: int = 14,
) -> Dict[str, Any]:
    """
    ReAct-style loop that processes ALL tool calls SEQUENTIALLY.

    This is a drop-in replacement for _run_react_json_agent that works
    with LLaMA 3 models which reject multiple parallel tool calls.

    How it works:
    - When LLM returns N tool calls in one response, we DON'T append
      that response directly to messages
    - Instead, we create N separate (AIMessage, ToolMessage) pairs
    - Each AIMessage contains only ONE tool call
    - This satisfies LLaMA 3's chat template constraint

    Args:
        llm: The LangChain LLM instance (will have tools bound)
        system_prompt: System prompt for the agent
        user_payload: Initial user message payload (will be JSON serialized)
        tools: List of LangChain tools to bind
        max_rounds: Maximum number of LLM invocations (not tool calls)

    Returns:
        Parsed JSON dict from the final LLM response
    """
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    messages: List[BaseMessage] = [
        SystemMessage(system_prompt),
        HumanMessage(content=json.dumps(user_payload, indent=2)),
    ]

    for round_num in range(max_rounds):
        resp = llm_with_tools.invoke(messages)
        tool_calls = getattr(resp, "tool_calls", None) or []

        # Check for invalid_tool_calls (vLLM returns these when parser fails)
        invalid_tcs = getattr(resp, "invalid_tool_calls", None) or []
        if invalid_tcs:
            logger.warning(
                f"Round {round_num + 1}: Model produced {len(invalid_tcs)} invalid tool call(s). "
                "Asking model to retry with correct format."
            )
            messages.append(resp)
            messages.append(
                HumanMessage(
                    content=(
                        "Your last tool call was malformed and could not be parsed. "
                        "You must include the 'name' key. Correct format:\n"
                        '{"name": "<tool_name>", "parameters": {<args>}}\n'
                        f"Available tools: {list(tool_map.keys())}. Please try again."
                    )
                )
            )
            continue

        if not tool_calls:
            # Check if content looks like a failed tool call missing 'name'
            content = getattr(resp, "content", "") or ""
            if content:
                try:
                    maybe_tc = json.loads(content)
                    if isinstance(maybe_tc, dict) and "parameters" in maybe_tc and "name" not in maybe_tc:
                        logger.warning(
                            f"Round {round_num + 1}: Content looks like a tool call missing 'name'. "
                            "Asking model to retry."
                        )
                        messages.append(resp)
                        messages.append(
                            HumanMessage(
                                content=(
                                    "Your response looks like a tool call but is missing the 'name' key. "
                                    "Correct format:\n"
                                    '{"name": "<tool_name>", "parameters": {<args>}}\n'
                                    f"Available tools: {list(tool_map.keys())}. Please try again."
                                )
                            )
                        )
                        continue
                except (json.JSONDecodeError, TypeError):
                    pass
            # No tool calls - this is the final answer
            return _extract_json(content)

        # Process each tool call as a SEPARATE conversation turn
        # This avoids LLaMA 3's "single tool-calls at once" restriction
        if len(tool_calls) > 1:
            logger.info(
                f"Round {round_num + 1}: LLM requested {len(tool_calls)} tool calls. "
                f"Processing sequentially: {[tc.get('name') for tc in tool_calls]}"
            )

        for i, tc in enumerate(tool_calls):
            name = tc.get("name")
            args = tc.get("args", {}) or {}
            # Handle nested properties format (LLaMA sometimes wraps args in 'properties')
            if "properties" in args and isinstance(args["properties"], dict):
                args = args["properties"]
            tool_id = tc.get("id")

            if name not in tool_map:
                raise ValueError(f"Unknown tool requested by model: {name}")

            # Create a NEW AIMessage with ONLY this single tool call
            # This is the key fix - each tool call gets its own AIMessage
            single_tool_ai_msg = AIMessage(
                content="",  # Tool-calling messages typically have empty content
                tool_calls=[
                    {
                        "name": name,
                        "args": args,
                        "id": tool_id,
                        "type": "tool_call",
                    }
                ],
            )
            messages.append(single_tool_ai_msg)

            # Execute the tool
            logger.debug(f"Executing tool {name} with args: {args}")
            result = tool_map[name].invoke(args)

            # Add the corresponding ToolMessage
            messages.append(
                ToolMessage(
                    content=json.dumps(
                        {"tool": name, "args": args, "result": result},
                        default=str,
                    ),
                    tool_call_id=tool_id,
                )
            )

        # After processing all tool calls, continue to next LLM invocation
        # The LLM will see all tool results and decide next action

    raise RuntimeError("Exceeded max rounds without producing a final JSON answer.")


# Backwards compatibility alias
run_react_single_tool = run_react_sequential_tools


def patch_agent_module():
    """
    Monkey-patch the agent module to use sequential-tool ReAct loop.

    Call this BEFORE running the agent workflow:

        from Agents.react_single_tool import patch_agent_module
        patch_agent_module()

        # Now run the agent
        from Agents.agent import app
        result = app.invoke(initial_state)
    """
    import Agents.agent as agent_module
    agent_module._run_react_json_agent = run_react_sequential_tools
    logger.info("Patched agent module to use sequential-tool ReAct loop")
