# Sharadar Data Dictionary

## SHARADAR_ACTIONS
**File:** `SHARADAR_ACTIONS_2_d1d74f09961f5a075917e43ce5da531e.csv`

Corporate actions data.

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

## SHARADAR_DAILY
**File:** `SHARADAR_DAILY_3_1c00e922d0fc2ccdfae0e4c5271349a4.csv`

Daily valuation metrics and ratios.

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

## SHARADAR_EVENTS
**File:** `SHARADAR_EVENTS_2_07fc6fea4b8b75a39306b896245e11e9.csv`

Corporate event codes by ticker and date.

| Column | Description |
|--------|-------------|
| ticker | Ticker symbol |
| date | Date of the event |
| eventcodes | Coded event identifiers |

---

## SHARADAR_INDICATORS
**Files:**
- `SHARADAR_INDICATORS_2_be12e8d4f43d8cbd38ad9e8c439b60a3.csv`
- `SHARADAR_INDICATORS_2_f515cff55958035cf2031ca28bc02fab.csv`

Metadata table describing indicators across all Sharadar tables.

| Column | Description |
|--------|-------------|
| table | Source table name |
| indicator | Indicator/column name |
| isfilter | Whether the indicator can be used as a filter |
| isprimarykey | Whether the indicator is part of the primary key |
| title | Human-readable title |
| description | Detailed description of the indicator |
| unittype | Unit of measurement (e.g., USD, ratio, percentage) |

---

## SHARADAR_METRICS
**File:** `SHARADAR_METRICS_a5c0f19b4a6d6a0eb7d8a2c89c55071b.csv`

Derived market metrics and technical indicators.

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

## SHARADAR_SEP
**File:** `SHARADAR_SEP_2_0afbc06bfa7d2d5ebd28c43e0940ec30.csv`

Daily stock prices (equity prices).

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

## SHARADAR_SF1
**File:** `SHARADAR_SF1_3_15e102612341e619fb9bbd8f089942ea.csv`

Core fundamental data (income statement, balance sheet, cash flow).

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

## SHARADAR_SF2
**File:** `SHARADAR_SF2_2_4d7bb742c3823b4926230eee3d6f0d53.csv`

Insider transactions (SEC Forms 3, 4, 5).

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

## SHARADAR_SF3A
**File:** `SHARADAR_SF3A_2_e74ba2dcabe9995b3a75dbdd035c774b.csv`

Institutional ownership aggregated by ticker.

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

## SHARADAR_SF3B
**File:** `SHARADAR_SF3B_2_dd60617cea0d0350a9bb47aac270a184.csv`

Institutional ownership aggregated by investor.

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

## SHARADAR_SF3
**File:** `SHARADAR_SF3_2_4d1961ca5b0d2fcbb4851fd57f82f461.csv`

Institutional ownership detail (per ticker, investor, security type).

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

## SHARADAR_SFP
**File:** `SHARADAR_SFP_2_fb4f5d2244276f3cfeca03f46b122d99.csv`

Daily stock prices (fund prices — ETFs, mutual funds, etc.).

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

## SHARADAR_SP500
**File:** `SHARADAR_SP500_2_d3dd60174f12d95cc9c92159fcbe44f1.csv`

S&P 500 index membership changes.

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

## SHARADAR_TICKERS
**File:** `SHARADAR_TICKERS_2_7ef16c1ffc4112bb028c13f3e3743955.csv`

Ticker metadata and company information.

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
