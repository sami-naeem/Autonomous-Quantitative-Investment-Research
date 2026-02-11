# Sharadar Data Dictionary

## Quick Reference: Table Routing Guide

Use this section to determine which table(s) to query based on the task at hand.

### Decision Tree

| If you need... | Use table | Join with |
|----------------|-----------|-----------|
| Daily OHLCV prices for **stocks** | **SEP** | TICKERS for metadata |
| Daily OHLCV prices for **ETFs/funds** | **SFP** | TICKERS for metadata |
| Valuation ratios (P/E, P/B, EV/EBITDA) on a daily basis | **DAILY** | SEP or SFP for prices |
| Technical indicators (moving averages, beta, returns) | **METRICS** | SEP for raw prices |
| Income statement, balance sheet, cash flow data | **SF1** | TICKERS for sector/industry |
| Insider buying/selling (SEC Forms 3/4/5) | **SF2** | SF1 for fundamentals context |
| Institutional ownership detail (who owns what) | **SF3** | SF3A or SF3B for aggregates |
| Institutional ownership aggregated **by ticker** | **SF3A** | TICKERS for metadata |
| Institutional ownership aggregated **by investor** | **SF3B** | SF3 for detail |
| Corporate actions (splits, dividends, M&A) | **ACTIONS** | SEP for price impact |
| Corporate events (earnings dates, etc.) | **EVENTS** | SF1 for earnings data |
| S&P 500 membership history (additions/removals) | **SP500** | TICKERS, SEP for analysis |
| Ticker metadata, sector, industry, exchange, delisting | **TICKERS** | Everything (universal lookup) |
| Column definitions and metadata for any table | **INDICATORS** | N/A (reference only) |

### Common Use Cases → Table Combinations

| Use Case | Primary Table | Supporting Tables |
|----------|---------------|-------------------|
| Screen stocks by fundamentals (e.g., low P/E, high ROE) | SF1 | TICKERS (sector filter), DAILY (market cap) |
| Backtest a price-based strategy | SEP (stocks) or SFP (funds) | TICKERS (universe filter), SP500 (index membership) |
| Analyze earnings surprise / post-earnings drift | SF1 | EVENTS (earnings dates), SEP (price reaction) |
| Track insider buying as a signal | SF2 | SEP (price), SF1 (fundamentals) |
| Identify institutional accumulation | SF3 or SF3A | SEP (price trends), TICKERS (metadata) |
| Build a dividend strategy | SF1 (dps, divyield, payoutratio) | ACTIONS (dividend events), METRICS (dividendyieldforward/trailing) |
| Analyze M&A or spin-off impact | ACTIONS | SEP (price), TICKERS (contra-party lookup) |
| Construct S&P 500 point-in-time universe | SP500 | SEP (prices), SF1 (fundamentals) |
| Compare valuation multiples across sectors | DAILY | TICKERS (sector/industry), SF1 (earnings context) |
| Risk analysis (beta, volatility) | METRICS | SEP (raw prices for custom calcs) |

---

## Table Summaries

### Relationships

All tables connect through `ticker` as the common join key. **TICKERS** is the universal dimension table — always join here for sector, industry, exchange, and delisting status. Time-series tables use `date` or `calendardate` as the time key.

```
TICKERS (dimension)
  ├── SEP (daily equity prices)           join on: ticker
  ├── SFP (daily fund prices)             join on: ticker
  ├── DAILY (daily valuation ratios)      join on: ticker, date
  ├── METRICS (technical indicators)      join on: ticker, date
  ├── SF1 (fundamentals)                  join on: ticker
  ├── SF2 (insider transactions)          join on: ticker
  ├── SF3 (institutional detail)          join on: ticker
  ├── SF3A (institutional by ticker)      join on: ticker
  ├── SF3B (institutional by investor)    join on: investorname → SF3
  ├── ACTIONS (corporate actions)         join on: ticker
  ├── EVENTS (corporate events)           join on: ticker
  └── SP500 (index membership)            join on: ticker

INDICATORS (metadata — describes columns in all other tables)
```

---

## Table Details

---

### SHARADAR_TICKERS *(Dimension Table — Start Here)*
**File:** `SHARADAR_TICKERS_2_7ef16c1ffc4112bb028c13f3e3743955.csv`

**Purpose:** Master ticker reference. Use to filter universes, look up sector/industry, check delisting status, and determine date coverage.

**Grain:** One row per ticker per source table (a ticker may appear in both SEP and SF1).

**Primary Key:** `table` + `permaticker`

**When to use:** Almost always. Join this to any other table to get company metadata. Use `isdelisted`, `category`, `sector`, `industry`, `scalemarketcap` to filter your universe before querying other tables. Check `firstpricedate`/`lastpricedate` and `firstquarter`/`lastquarter` to confirm data coverage.

**Key filters:** `isdelisted` (active vs. delisted), `category` (Domestic Common Stock, ETF, ADR, etc.), `exchange` (NYSE, NASDAQ, etc.), `sector`, `industry`, `scalemarketcap` (Nano/Micro/Small/Mid/Large/Mega)

| Column | Description |
|--------|-------------|
| table | Source table (SEP, SFP, etc.) |
| permaticker | Permanent ticker ID (survives ticker changes) |
| ticker | Current ticker symbol |
| name | Company name |
| exchange | Stock exchange |
| isdelisted | Whether the ticker is delisted |
| category | Security category (e.g., Domestic Common Stock, ETF) |
| cusips | CUSIP identifier(s) |
| siccode | SIC industry code |
| sicsector | SIC sector |
| sicindustry | SIC industry |
| famasector | Fama-French sector |
| famaindustry | Fama-French industry |
| sector | Sector classification |
| industry | Industry classification |
| scalemarketcap | Market cap scale (Nano, Micro, Small, Mid, Large, Mega) |
| scalerevenue | Revenue scale |
| relatedtickers | Related ticker symbols |
| currency | Reporting currency |
| location | Company headquarters location |
| lastupdated | Date the record was last updated |
| firstadded | Date the ticker was first added |
| firstpricedate | First available price date |
| lastpricedate | Last available price date |
| firstquarter | First available quarterly report |
| lastquarter | Last available quarterly report |
| secfilings | URL to SEC filings |
| companysite | Company website URL |

---

### SHARADAR_SEP *(Daily Equity Prices)*
**File:** `SHARADAR_SEP_2_0afbc06bfa7d2d5ebd28c43e0940ec30.csv`

**Purpose:** Daily OHLCV price data for individual equities (common stocks, ADRs, etc.).

**Grain:** One row per ticker per trading day.

**Primary Key:** `ticker` + `date`

**When to use:** Any price-based analysis for stocks. Use `closeadj` for backtesting (adjusted for splits and dividends). Use `closeunadj` for point-in-time display. For ETF/fund prices, use **SFP** instead.

| Column | Description |
|--------|-------------|
| ticker | Ticker symbol |
| date | Date |
| open | Opening price |
| high | Intraday high price |
| low | Intraday low price |
| close | Closing price |
| volume | Trading volume |
| closeadj | Split- and dividend-adjusted closing price |
| closeunadj | Unadjusted closing price |
| lastupdated | Date the record was last updated |

---

### SHARADAR_SFP *(Daily Fund Prices)*
**File:** `SHARADAR_SFP_2_fb4f5d2244276f3cfeca03f46b122d99.csv`

**Purpose:** Daily OHLCV price data for ETFs, mutual funds, and other fund vehicles.

**Grain:** One row per ticker per trading day.

**Primary Key:** `ticker` + `date`

**When to use:** Same as SEP but for funds/ETFs. Check TICKERS `category` field to determine if a ticker belongs in SEP or SFP.

| Column | Description |
|--------|-------------|
| ticker | Ticker symbol |
| date | Date |
| open | Opening price |
| high | Intraday high price |
| low | Intraday low price |
| close | Closing price |
| volume | Trading volume |
| closeadj | Split- and dividend-adjusted closing price |
| closeunadj | Unadjusted closing price |
| lastupdated | Date the record was last updated |

---

### SHARADAR_DAILY *(Daily Valuation Ratios)*
**File:** `SHARADAR_DAILY_3_1c00e922d0fc2ccdfae0e4c5271349a4.csv`

**Purpose:** Pre-computed daily valuation multiples. Saves you from having to join SF1 fundamentals to SEP prices and compute ratios yourself.

**Grain:** One row per ticker per trading day.

**Primary Key:** `ticker` + `date`

**When to use:** Screening or time-series analysis of valuation multiples (P/E, P/B, P/S, EV/EBIT, EV/EBITDA). Faster than computing from SF1 + SEP.

| Column | Description |
|--------|-------------|
| ticker | Ticker symbol |
| date | Date |
| lastupdated | Date the record was last updated |
| ev | Enterprise value |
| evebit | Enterprise value to EBIT ratio |
| evebitda | Enterprise value to EBITDA ratio |
| marketcap | Market capitalization |
| pb | Price-to-book ratio |
| pe | Price-to-earnings ratio |
| ps | Price-to-sales ratio |

---

### SHARADAR_METRICS *(Technical / Market Indicators)*
**File:** `SHARADAR_METRICS_a5c0f19b4a6d6a0eb7d8a2c89c55071b.csv`

**Purpose:** Pre-computed technical and market indicators: moving averages, beta, dividend yields, returns, volume averages.

**Grain:** One row per ticker per date.

**Primary Key:** `ticker` + `date`

**When to use:** Screening on technical criteria (e.g., price above 200-day MA), risk metrics (beta), momentum (1Y/5Y/YTD returns), or dividend analysis (forward/trailing yields). Saves computing these from raw SEP prices.

| Column | Description |
|--------|-------------|
| ticker | Ticker symbol |
| date | Date |
| lastupdated | Date the record was last updated |
| beta1y | 1-year beta |
| beta5y | 5-year beta |
| dividendyieldforward | Forward dividend yield |
| dividendyieldtrailing | Trailing dividend yield |
| high52w | 52-week high price |
| high5y | 5-year high price |
| low52w | 52-week low price |
| low5y | 5-year low price |
| ma200d | 200-day moving average |
| ma200w | 200-week moving average |
| ma50d | 50-day moving average |
| ma50w | 50-week moving average |
| price | Current price |
| return1y | 1-year return |
| return5y | 5-year return |
| returnytd | Year-to-date return |
| volume | Daily volume |
| volumeavg1m | 1-month average volume |
| volumeavg3m | 3-month average volume |

---

### SHARADAR_SF1 *(Core Fundamentals)*
**File:** `SHARADAR_SF1_3_15e102612341e619fb9bbd8f089942ea.csv`

**Purpose:** Comprehensive fundamental data covering income statement, balance sheet, and cash flow statement. The most detailed financial table in the dataset.

**Grain:** One row per ticker per dimension per reporting period.

**Primary Key:** `ticker` + `dimension` + `calendardate`

**When to use:** Any fundamental analysis — revenue, earnings, margins, cash flow, leverage, returns on capital, per-share metrics. This is the core table for financial modeling, screening, and factor construction.

**Critical: Understanding the `dimension` column:**

| Dimension | Meaning | Use For |
|-----------|---------|---------|
| ARQ | As-Reported Quarterly | Point-in-time quarterly filings |
| ARY | As-Reported Annual | Point-in-time annual filings |
| MRQ | Most-Recent Quarterly | Latest restated quarterly |
| MRY | Most-Recent Annual | Latest restated annual |
| ART | As-Reported Trailing-12-Months | TTM from as-reported quarters |
| MRT | Most-Recent Trailing-12-Months | TTM from restated quarters |

For backtesting, use **AR\*** dimensions (as-reported) to avoid look-ahead bias. For current analysis, **MR\*** dimensions reflect the latest restated numbers.

| Column | Description |
|--------|-------------|
| ticker | Ticker symbol |
| dimension | Reporting dimension (ARQ, ARY, MRQ, MRY, ART, MRT) |
| calendardate | Calendar date of the reporting period end |
| datekey | Date key for the filing |
| reportperiod | Report period end date |
| fiscalperiod | Fiscal period label (e.g., Q1, FY) |
| lastupdated | Date the record was last updated |
| accoci | Accumulated other comprehensive income |
| assets | Total assets |
| assetsavg | Average assets |
| assetsc | Current assets |
| assetsnc | Non-current assets |
| assetturnover | Asset turnover ratio |
| bvps | Book value per share |
| capex | Capital expenditures |
| cashneq | Cash and equivalents |
| cashnequsd | Cash and equivalents (USD) |
| cor | Cost of revenue |
| consolinc | Consolidated income |
| currentratio | Current ratio |
| de | Debt-to-equity ratio |
| debt | Total debt |
| debtc | Current debt |
| debtnc | Non-current debt |
| debtusd | Total debt (USD) |
| deferredrev | Deferred revenue |
| depamor | Depreciation and amortization |
| deposits | Deposits |
| divyield | Dividend yield |
| dps | Dividends per share |
| ebit | Earnings before interest and taxes |
| ebitda | Earnings before interest, taxes, depreciation, and amortization |
| ebitdamargin | EBITDA margin |
| ebitdausd | EBITDA (USD) |
| ebitusd | EBIT (USD) |
| ebt | Earnings before taxes |
| eps | Earnings per share (basic) |
| epsdil | Earnings per share (diluted) |
| epsusd | Earnings per share (USD) |
| equity | Total equity |
| equityavg | Average equity |
| equityusd | Total equity (USD) |
| ev | Enterprise value |
| evebit | EV/EBIT ratio |
| evebitda | EV/EBITDA ratio |
| fcf | Free cash flow |
| fcfps | Free cash flow per share |
| fxusd | Foreign exchange rate to USD |
| gp | Gross profit |
| grossmargin | Gross margin |
| intangibles | Intangible assets |
| intexp | Interest expense |
| invcap | Invested capital |
| invcapavg | Average invested capital |
| inventory | Inventory |
| investments | Total investments |
| investmentsc | Current investments |
| investmentsnc | Non-current investments |
| liabilities | Total liabilities |
| liabilitiesc | Current liabilities |
| liabilitiesnc | Non-current liabilities |
| marketcap | Market capitalization |
| ncf | Net cash flow |
| ncfbus | Cash flow from business acquisitions/disposals |
| ncfcommon | Cash flow from issuance/repurchase of common stock |
| ncfdebt | Cash flow from issuance/repayment of debt |
| ncfdiv | Cash flow from dividends paid |
| ncff | Net cash flow from financing |
| ncfi | Net cash flow from investing |
| ncfinv | Cash flow from investment acquisitions/disposals |
| ncfo | Net cash flow from operations |
| ncfx | Effect of exchange rate changes on cash |
| netinc | Net income |
| netinccmn | Net income common |
| netinccmnusd | Net income common (USD) |
| netincdis | Net income from discontinued operations |
| netincnci | Net income attributable to non-controlling interests |
| netmargin | Net margin |
| opex | Operating expenses |
| opinc | Operating income |
| payables | Accounts payable |
| payoutratio | Payout ratio |
| pb | Price-to-book ratio |
| pe | Price-to-earnings ratio |
| pe1 | Price-to-earnings ratio (1-year) |
| ppnenet | Property, plant, and equipment (net) |
| prefdivis | Preferred dividends |
| price | Share price |
| ps | Price-to-sales ratio |
| ps1 | Price-to-sales ratio (1-year) |
| receivables | Accounts receivable |
| retearn | Retained earnings |
| revenue | Total revenue |
| revenueusd | Total revenue (USD) |
| rnd | Research and development expense |
| roa | Return on assets |
| roe | Return on equity |
| roic | Return on invested capital |
| ros | Return on sales |
| sbcomp | Stock-based compensation |
| sgna | Selling, general, and administrative expense |
| sharefactor | Share factor (adjustment for splits) |
| sharesbas | Shares outstanding (basic) |
| shareswa | Weighted average shares outstanding |
| shareswadil | Weighted average shares outstanding (diluted) |
| sps | Sales per share |
| tangibles | Tangible assets |
| taxassets | Tax assets |
| taxexp | Tax expense |
| taxliabilities | Tax liabilities |
| tbvps | Tangible book value per share |
| workingcapital | Working capital |

---

### SHARADAR_SF2 *(Insider Transactions)*
**File:** `SHARADAR_SF2_2_4d7bb742c3823b4926230eee3d6f0d53.csv`

**Purpose:** Insider buy/sell transactions from SEC Forms 3, 4, and 5.

**Grain:** One row per transaction per insider per filing.

**Primary Key:** `ticker` + `filingdate` + `ownername` + `rownum`

**When to use:** Insider sentiment analysis. Filter on `transactioncode` (P = open-market purchase, S = open-market sale) and `isdirector`/`isofficer`/`istenpercentowner` for signal quality. Large cluster purchases by multiple insiders can be a bullish signal.

| Column | Description |
|--------|-------------|
| ticker | Ticker symbol |
| filingdate | Date the SEC filing was made |
| formtype | SEC form type (3, 4, or 5) |
| issuername | Name of the issuing company |
| ownername | Name of the insider |
| officertitle | Title of the officer |
| isdirector | Whether the insider is a director |
| isofficer | Whether the insider is an officer |
| istenpercentowner | Whether the insider is a 10%+ owner |
| transactiondate | Date of the transaction |
| securityadcode | Security acquisition/disposition code |
| transactioncode | Transaction type code |
| sharesownedbeforetransaction | Shares owned before the transaction |
| transactionshares | Number of shares transacted |
| sharesownedfollowingtransaction | Shares owned after the transaction |
| transactionpricepershare | Price per share in the transaction |
| transactionvalue | Total value of the transaction |
| securitytitle | Title/type of the security |
| directorindirect | Direct or indirect ownership |
| natureofownership | Nature of indirect ownership |
| dateexercisable | Date options become exercisable |
| priceexercisable | Exercise price of options |
| expirationdate | Expiration date of options |
| rownum | Row number identifier |

---

### SHARADAR_SF3 *(Institutional Ownership — Detail)*
**File:** `SHARADAR_SF3_2_4d1961ca5b0d2fcbb4851fd57f82f461.csv`

**Purpose:** Granular institutional ownership: which investor holds what security type in which company, with units and value.

**Grain:** One row per ticker per investor per security type per quarter.

**Primary Key:** `ticker` + `investorname` + `securitytype` + `calendardate`

**When to use:** Drill into specific institutional positions. Use SF3A for ticker-level aggregates or SF3B for investor-level aggregates instead if you don't need per-investor-per-ticker detail.

| Column | Description |
|--------|-------------|
| ticker | Ticker symbol |
| investorname | Name of the institutional investor |
| securitytype | Type of security held |
| calendardate | Calendar date of the reporting period |
| value | Value of the holding |
| units | Number of units held |
| price | Price per unit |

---

### SHARADAR_SF3A *(Institutional Ownership — By Ticker)*
**File:** `SHARADAR_SF3A_2_e74ba2dcabe9995b3a75dbdd035c774b.csv`

**Purpose:** Institutional ownership aggregated at the ticker level. Shows how many institutions hold a stock, total units, and total value across security types.

**Grain:** One row per ticker per quarter.

**Primary Key:** `ticker` + `calendardate`

**When to use:** Quick lookup of institutional ownership breadth and concentration for a given stock. Use this for screening (e.g., "stocks with >100 institutional holders"). For investor-level view, use SF3B. For detail, use SF3.

| Column | Description |
|--------|-------------|
| calendardate | Calendar date of the reporting period |
| ticker | Ticker symbol |
| name | Company name |
| shrholders | Number of share holders |
| cllholders | Number of call option holders |
| putholders | Number of put option holders |
| wntholders | Number of warrant holders |
| dbtholders | Number of debt holders |
| prfholders | Number of preferred stock holders |
| fndholders | Number of fund holders |
| undholders | Number of undefined security holders |
| shrunits | Total share units held |
| cllunits | Total call option units held |
| putunits | Total put option units held |
| wntunits | Total warrant units held |
| dbtunits | Total debt units held |
| prfunits | Total preferred stock units held |
| fndunits | Total fund units held |
| undunits | Total undefined security units held |
| shrvalue | Total value of shares held |
| cllvalue | Total value of call options held |
| putvalue | Total value of put options held |
| wntvalue | Total value of warrants held |
| dbtvalue | Total value of debt held |
| prfvalue | Total value of preferred stock held |
| fndvalue | Total value of fund holdings |
| undvalue | Total value of undefined securities held |
| totalvalue | Total value of all holdings |
| percentoftotal | Percent of total institutional ownership |

---

### SHARADAR_SF3B *(Institutional Ownership — By Investor)*
**File:** `SHARADAR_SF3B_2_dd60617cea0d0350a9bb47aac270a184.csv`

**Purpose:** Institutional ownership aggregated at the investor level. Shows total portfolio breadth and value for each institutional investor.

**Grain:** One row per investor per quarter.

**Primary Key:** `investorname` + `calendardate`

**When to use:** Analyze a specific institution's total holdings (e.g., "What is Berkshire Hathaway's total portfolio?"). For which stocks an institution holds, use SF3.

| Column | Description |
|--------|-------------|
| calendardate | Calendar date of the reporting period |
| investorname | Name of the institutional investor |
| shrholdings | Number of share holdings |
| cllholdings | Number of call option holdings |
| putholdings | Number of put option holdings |
| wntholdings | Number of warrant holdings |
| dbtholdings | Number of debt holdings |
| prfholdings | Number of preferred stock holdings |
| fndholdings | Number of fund holdings |
| undholdings | Number of undefined security holdings |
| shrunits | Total share units held |
| cllunits | Total call option units held |
| putunits | Total put option units held |
| wntunits | Total warrant units held |
| dbtunits | Total debt units held |
| prfunits | Total preferred stock units held |
| fndunits | Total fund units held |
| undunits | Total undefined security units held |
| shrvalue | Total value of shares held |
| cllvalue | Total value of call options held |
| putvalue | Total value of put options held |
| wntvalue | Total value of warrants held |
| dbtvalue | Total value of debt held |
| prfvalue | Total value of preferred stock held |
| fndvalue | Total value of fund holdings |
| undvalue | Total value of undefined securities held |
| totalvalue | Total value of all holdings |
| percentoftotal | Percent of total institutional ownership |

---

### SHARADAR_ACTIONS *(Corporate Actions)*
**File:** `SHARADAR_ACTIONS_2_d1d74f09961f5a075917e43ce5da531e.csv`

**Purpose:** Corporate actions including stock splits, dividends, spinoffs, acquisitions, and other events that affect share structure or ownership.

**Grain:** One row per action per ticker per date.

**Primary Key:** `ticker` + `date` + `action`

**When to use:** Adjusting for splits in custom calculations, tracking dividend history, identifying M&A activity, or understanding corporate events that impact price. The `contraticker`/`contraname` fields identify the counterparty in M&A transactions.

| Column | Description |
|--------|-------------|
| date | Date of the corporate action |
| action | Type of action (e.g., split, dividend, acquisition) |
| ticker | Ticker symbol of the company |
| name | Company name |
| value | Value associated with the action (e.g., split ratio, dividend amount) |
| contraticker | Ticker of the counterparty (e.g., acquirer/target) |
| contraname | Name of the counterparty |

---

### SHARADAR_EVENTS *(Corporate Events)*
**File:** `SHARADAR_EVENTS_2_07fc6fea4b8b75a39306b896245e11e9.csv`

**Purpose:** Coded corporate events such as earnings dates, FDA approvals, conferences, and other calendar events.

**Grain:** One row per ticker per date.

**Primary Key:** `ticker` + `date`

**When to use:** Building event-driven strategies, identifying earnings dates for post-earnings drift analysis, or filtering time periods around corporate events.

| Column | Description |
|--------|-------------|
| ticker | Ticker symbol |
| date | Date of the event |
| eventcodes | Coded event identifiers |

---

### SHARADAR_SP500 *(S&P 500 Membership)*
**File:** `SHARADAR_SP500_2_d3dd60174f12d95cc9c92159fcbe44f1.csv`

**Purpose:** Historical record of S&P 500 index additions and removals. Essential for constructing point-in-time index membership to avoid survivorship bias.

**Grain:** One row per membership change event.

**Primary Key:** `ticker` + `date` + `action`

**When to use:** Constructing a survivorship-bias-free S&P 500 universe for backtesting. Replay additions/removals chronologically to know which stocks were in the index at any given date. The `contraticker` field shows which company was swapped in/out.

| Column | Description |
|--------|-------------|
| date | Date of the membership change |
| action | Action type (added/removed) |
| ticker | Ticker symbol |
| name | Company name |
| contraticker | Ticker of the company being replaced/replacing |
| contraname | Name of the company being replaced/replacing |
| note | Additional notes |

---

### SHARADAR_INDICATORS *(Metadata / Column Definitions)*
**Files:**
- `SHARADAR_INDICATORS_2_be12e8d4f43d8cbd38ad9e8c439b60a3.csv`
- `SHARADAR_INDICATORS_2_f515cff55958035cf2031ca28bc02fab.csv`

**Purpose:** Self-describing metadata table. Contains human-readable descriptions, unit types, and key designations for every column in every other Sharadar table.

**Grain:** One row per indicator per table.

**Primary Key:** `table` + `indicator`

**When to use:** Programmatic lookup of column meanings. An agent can query this table to understand any column in any other table without needing this data dictionary. Also useful for building dynamic UIs or documentation.

**Note:** Two files exist with different hashes — verify if they contain different content or are duplicates.

| Column | Description |
|--------|-------------|
| table | Source table name |
| indicator | Indicator/column name |
| isfilter | Whether the indicator can be used as a filter |
| isprimarykey | Whether the indicator is part of the primary key |
| title | Human-readable title |
| description | Detailed description of the indicator |
| unittype | Unit of measurement (e.g., USD, ratio, percentage) |
