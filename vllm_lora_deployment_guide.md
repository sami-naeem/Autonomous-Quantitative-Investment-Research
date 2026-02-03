# vLLM LoRA Deployment Guide

Deploy your fine-tuned Llama 3.1 8B model with LoRA adapters on Lambda Labs.

------------------------------------------------------------------------

## Overview

| Component    | Value                                      |
|--------------|--------------------------------------------|
| Base Model   | `meta-llama/Llama-3.1-8B-Instruct`         |
| LoRA Adapter | `finllama` (quantitative finance research) |
| Server       | vLLM v0.15.0                               |
| API          | OpenAI-compatible                          |
| GPU          | Lambda Labs A100/H100                      |

------------------------------------------------------------------------

## Quick Start

### 1. SSH into Lambda Instance

``` bash
ssh ubuntu@<LAMBDA_IP>
```

### 2. Run Setup Script

``` bash
bash setup_lambda_lora.sh
```

This will: - Install vLLM and dependencies - Download Llama 3.1 8B base model - Create directory structure - Generate startup scripts

### 3. Upload LoRA Adapters (from your Local)

``` bash
scp -r "/path/to/lora_adapters/"* ubuntu@<LAMBDA_IP>:~/lora_adapters/finllama-lora/
```

**Required files:**

```         
~/lora_adapters/finllama-lora/
├── adapter_config.json
├── adapter_model.safetensors
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
└── README.md
```

### 4. Start vLLM Server

``` bash
~/scripts/start_vllm_lora.sh
```

Wait for: `Application startup complete.`

------------------------------------------------------------------------

## API Usage

### Endpoint

```         
http://<LAMBDA_IP>:8000/v1/chat/completions
```

### Authentication

```         
Authorization: Bearer token-abc123
```

### Available Models

| Model Name                         | Description           |
|------------------------------------|-----------------------|
| `finllama`                         | LoRA fine-tuned model |
| `/home/ubuntu/models/llama-3.1-8b` | Base model (no LoRA)  |

------------------------------------------------------------------------

## Example API Calls

### List Models

``` bash
curl http://<LAMBDA_IP>:8000/v1/models \
  -H "Authorization: Bearer token-abc123"
```

### Basic Completion

``` bash
curl http://<LAMBDA_IP>:8000/v1/chat/completions \
  -H "Authorization: Bearer token-abc123" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "finllama",
    "max_tokens": 1024,
    "messages": [
      {"role": "user", "content": "How can machine learning improve options pricing?"}
    ]
  }'
```

### With System Prompt (Recommended)

``` bash
curl http://<LAMBDA_IP>:8000/v1/chat/completions \
  -H "Authorization: Bearer token-abc123" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "finllama",
    "max_tokens": 1024,
    "temperature": 0.7,
    "repetition_penalty": 1.1,
    "messages": [
      {
        "role": "system",
        "content": "You are an expert quantitative finance researcher. Given a research question, develop a comprehensive research plan with: 1. Problem formulation 2. Mathematical framework 3. Methodology 4. Data requirements 5. Expected contributions"
      },
      {
        "role": "user",
        "content": "How can reinforcement learning be applied to dynamic portfolio rebalancing?"
      }
    ]
  }'
```

### Recommended Parameters

| Parameter            | Value | Purpose                 |
|----------------------|-------|-------------------------|
| `max_tokens`         | 1024  | Limit response length   |
| `temperature`        | 0.7   | Control randomness      |
| `repetition_penalty` | 1.1   | Prevent repetitive text |

------------------------------------------------------------------------

## Python Client

### Using OpenAI SDK

``` python
from openai import OpenAI

client = OpenAI(
    base_url="http://<LAMBDA_IP>:8000/v1",
    api_key="token-abc123"
)

response = client.chat.completions.create(
    model="finllama",
    max_tokens=1024,
    temperature=0.7,
    messages=[
        {
            "role": "system",
            "content": "You are an expert quantitative finance researcher."
        },
        {
            "role": "user",
            "content": "Develop a research plan for using NLP on earnings calls."
        }
    ],
    extra_body={
        "repetition_penalty": 1.1
    }
)

print(response.choices[0].message.content)
```

### Using Requests

``` python
import requests

url = "http://<LAMBDA_IP>:8000/v1/chat/completions"
headers = {
    "Authorization": "Bearer token-abc123",
    "Content-Type": "application/json"
}

payload = {
    "model": "finllama",
    "max_tokens": 1024,
    "temperature": 0.7,
    "repetition_penalty": 1.1,
    "messages": [
        {"role": "system", "content": "You are an expert quantitative finance researcher."},
        {"role": "user", "content": "How can graph neural networks model financial contagion?"}
    ]
}

response = requests.post(url, headers=headers, json=payload)
print(response.json()["choices"][0]["message"]["content"])
```

------------------------------------------------------------------------

## LangChain Integration

``` python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://<LAMBDA_IP>:8000/v1",
    api_key="token-abc123",
    model="finllama",
    max_tokens=1024,
    temperature=0.7,
    model_kwargs={"repetition_penalty": 1.1}
)

response = llm.invoke("Develop a research plan for crypto market microstructure.")
print(response.content)
```

------------------------------------------------------------------------

## Directory Structure on Lambda

```         
~/
├── models/
│   └── llama-3.1-8b/          # Base model weights
├── lora_adapters/
│   └── finllama-lora/         # Your LoRA adapter
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── special_tokens_map.json
├── scripts/
│   ├── start_vllm_lora.sh     # Start with LoRA
│   ├── start_vllm_base.sh     # Start base model only
│   ├── start_vllm_merged.sh   # Start merged model
│   └── merge_lora.py          # Merge LoRA into base
└── vllm_env/                  # Python virtual environment
```

------------------------------------------------------------------------

## Startup Scripts

### `start_vllm_lora.sh` - Run with LoRA (Dynamic Loading)

``` bash
~/scripts/start_vllm_lora.sh
```

-   Loads base model + LoRA adapter at runtime
-   Use `model="finllama"` in API calls
-   Slightly higher memory usage

### `start_vllm_base.sh` - Run Base Model Only

``` bash
~/scripts/start_vllm_base.sh
```

-   No LoRA, just the base Llama 3.1 8B
-   Use `model="/home/ubuntu/models/llama-3.1-8b"` in API calls

### `start_vllm_merged.sh` - Run Merged Model

First merge the LoRA weights:

``` bash
source ~/vllm_env/bin/activate
python ~/scripts/merge_lora.py
```

Then run:

``` bash
~/scripts/start_vllm_merged.sh
```

-   Faster inference (no LoRA overhead)
-   Use `model="/home/ubuntu/models/finllama-merged"` in API calls

------------------------------------------------------------------------

## Troubleshooting

### Connection Refused

``` bash
# Check if vLLM is running
ps aux | grep vllm

# Check logs
cat ~/vllm.log
```

### Out of Memory

Edit `~/scripts/start_vllm_lora.sh`:

``` bash
--gpu-memory-utilization 0.85  # Reduce from 0.90
--max-model-len 4096           # Reduce from 8192
```

### Repetitive Output

Add these parameters to your API calls:

``` json
{
  "repetition_penalty": 1.15,
  "temperature": 0.8,
  "max_tokens": 1024
}
```

### SCP Upload Fails

Use rsync instead (auto-resumes):

``` bash
rsync -avz --progress "/path/to/lora_adapters/" ubuntu@<LAMBDA_IP>:~/lora_adapters/finllama-lora/
```

Or compress first:

``` bash
# Local
tar -czvf lora_adapters.tar.gz lora_adapters/
scp lora_adapters.tar.gz ubuntu@<LAMBDA_IP>:~/

# On Lambda
tar -xzvf lora_adapters.tar.gz
mv lora_adapters/* ~/lora_adapters/finllama-lora/
```

------------------------------------------------------------------------

## Server Management

### Run in Background

``` bash
nohup ~/scripts/start_vllm_lora.sh > ~/vllm.log 2>&1 &
```

### View Logs

``` bash
tail -f ~/vllm.log
```

### Stop Server

``` bash
pkill -f vllm
```

### Check GPU Usage

``` bash
nvidia-smi
```

------------------------------------------------------------------------

## Sample Research Questions

Your model was fine-tuned for quantitative finance research planning. Try these:

1.  "How can machine learning improve options pricing models?"
2.  "How can reinforcement learning be applied to dynamic portfolio rebalancing under transaction costs?"
3.  "What is the impact of high-frequency trading on market liquidity and price discovery?"
4.  "How can NLP techniques extract alpha signals from earnings call transcripts?"
5.  "How can graph neural networks model systemic risk and financial contagion?"

------------------------------------------------------------------------

## Current Deployment

| Setting   | Value            |
|-----------|------------------|
| Lambda IP | `129.146.36.147` |
| Port      | `8000`           |
| API Key   | `token-abc123`   |
| Model     | `finllama`       |

**Test command:**

``` bash
curl http://129.146.36.147:8000/v1/chat/completions \
  -H "Authorization: Bearer token-abc123" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "finllama",
    "max_tokens": 1024,
    "temperature": 0.7,
    "repetition_penalty": 1.1,
    "messages": [
      {"role": "system", "content": "You are an expert quantitative finance researcher."},
      {"role": "user", "content": "How can machine learning improve options pricing?"}
    ]
  }'
```
