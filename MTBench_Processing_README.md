# MTBench Dataset Processing Scripts

This directory contains scripts to process MTBench weather and finance datasets into the format required for time series forecasting with text prompts.

## Generated Files

The scripts generate CSV files with the following naming convention:
- `MTBench_weather_{hist_len}_{pred_len}_{split}.csv`
- `MTBench_finance_{hist_len}_{pred_len}_{split}.csv`

Where:
- `hist_len`: Historical sequence length (e.g., 96)
- `pred_len`: Prediction sequence length (e.g., 12, 24, 48)
- `split`: Data split (train, val, test)

## Scripts

### 1. get_MTBench_weather_with_prompt.py
Processes weather data from `/data2/user2/MTBench/weather/`

**Usage:**
```bash
python get_MTBench_weather_with_prompt.py --hist_len 96 --pred_len 12
python get_MTBench_weather_with_prompt.py --hist_len 96 --pred_len 24
python get_MTBench_weather_with_prompt.py --hist_len 96 --pred_len 48
```

### 2. get_MTBench_finance_with_prompt.py
Processes finance data from `/data2/user2/MTBench/finance/`

**Usage:**
```bash
python get_MTBench_finance_with_prompt.py --hist_len 96 --pred_len 12
python get_MTBench_finance_with_prompt.py --hist_len 96 --pred_len 24
python get_MTBench_finance_with_prompt.py --hist_len 96 --pred_len 48
```

## Output Format

Each generated CSV file contains 4 columns:
1. `history_series`: List of historical time series values
2. `prompt`: Formatted text prompt with historical data and statistics
3. `horizon_series`: List of future prediction target values
4. `original_text`: Associated text information

## Features

The scripts feature:
- Processing all available time series data from MTBench
- Using sample texts for consistent text information across samples
- Generating complete datasets without sampling limitations
- Efficient data processing and memory management

## Output Location

All generated files are saved to: `/data2/user2/ter_tsf/processed_data/`

## Successfully Generated Files

- MTBench_weather_96_12_{train,val,test}.csv
- MTBench_weather_96_24_{train,val,test}.csv
- MTBench_weather_96_48_{train,val,test}.csv
- MTBench_finance_96_12_{train,val,test}.csv
- MTBench_finance_96_24_{train,val,test}.csv
- MTBench_finance_96_48_{train,val,test}.csv
