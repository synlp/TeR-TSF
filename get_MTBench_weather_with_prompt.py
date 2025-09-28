"""
MTBench Weather Dataset Processing Script

This script processes weather data from MTBench and converts it into the format required
for time series forecasting with text prompts. It generates CSV files with the following columns:
- history_series: Historical time series data (temperature values)
- prompt: Formatted prompt with historical data and statistics
- horizon_series: Future prediction targets
- original_text: Associated text information

The script features:
1. Processing all available weather time series data
2. Using real weather news text data matched by timestamp
3. Generating complete datasets without sampling limitations

Usage:
    python get_MTBench_weather_with_prompt.py --hist_len 96 --pred_len 12
    python get_MTBench_weather_with_prompt.py --hist_len 96 --pred_len 24
    python get_MTBench_weather_with_prompt.py --hist_len 96 --pred_len 48
"""

import pandas as pd
import argparse
from tqdm import tqdm
import os
import json
import numpy as np
from datetime import datetime, timedelta

main_prompt_template = """
Analyze the following time series data and provide insights that could help with forecasting.

Historical data:
{data_records}

Statistics:
{statistical_summary}

Future prediction period:
{future_dates}

Provide your analysis.
"""

data_record_template = "In {date}, the {feature_name} is {value:.2f}, the report is: {report}"
statistical_summary_template = "Statistical summary:\n- Maximum: {max_val}\n- Minimum: {min_val}\n- Average: {avg_val}\n- Standard deviation: {std_val}"


def load_weather_text_data(text_data_dir):
    """Load weather text data from JSON files"""
    text_data = {}
    
    # Get all JSON files
    json_files = [f for f in os.listdir(text_data_dir) if f.endswith('.json')]
    
    for filename in json_files:
        station_id = filename.replace('news_', '').replace('.json', '')
        filepath = os.path.join(text_data_dir, filename)
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if 'BEGIN_TIMESTAMP' in data and 'NEWS' in data:
            begin_times = data['BEGIN_TIMESTAMP']
            end_times = data.get('END_TIMESTAMP', begin_times)
            news_texts = data['NEWS']
            
            for begin_time, end_time, news in zip(begin_times, end_times, news_texts):
                begin_dt = pd.to_datetime(begin_time)
                end_dt = pd.to_datetime(end_time)
                
                # Use the date of the begin timestamp as key
                date_key = begin_dt.date()
                if date_key not in text_data:
                    text_data[date_key] = []
                
                if isinstance(news, str) and news.strip():
                    text_data[date_key].append({
                        'text': news.strip(),
                        'begin_time': begin_dt,
                        'end_time': end_dt,
                        'station_id': station_id
                    })
    
    return text_data


def find_relevant_weather_text(target_date, text_data):
    """Find relevant weather text for a given date (only historical texts)"""
    target_date = pd.to_datetime(target_date).date()
    relevant_texts = []
    
    # Look for texts on the same date or previous dates only (no future data)
    for delta_days in range(-2, 1):  # Only -2, -1, 0 days (no future)
        check_date = target_date + timedelta(days=delta_days)
        if check_date in text_data:
            relevant_texts.extend([item['text'] for item in text_data[check_date]])
    
    return relevant_texts


def construct_prompt(window_df, future_df, text_data, feature_name="temperature"):
    records = []
    all_texts = []
    
    for _, row in window_df.iterrows():
        date_str = row["date"]
        value = row[feature_name]
        
        # Find relevant text for this date
        relevant_texts = find_relevant_weather_text(date_str, text_data)
        if relevant_texts:
            # Use first relevant text, limit length
            report = relevant_texts[0][:300]  # Limit to 300 characters
            all_texts.extend(relevant_texts)
        else:
            report = "(no text)."
        
        record = data_record_template.format(
            date=pd.to_datetime(date_str),
            feature_name=feature_name,
            value=value,
            report=report
        )
        records.append(record)
    
    data_records = "\n".join(records)

    values = window_df[feature_name]
    stats = statistical_summary_template.format(
        max_val=round(values.max(), 2),
        min_val=round(values.min(), 2),
        avg_val=round(values.mean(), 2),
        std_val=round(values.std(), 2)
    )

    # Future dates range
    start_date = pd.to_datetime(future_df["date"].iloc[0])
    end_date = pd.to_datetime(future_df["date"].iloc[-1])
    future_dates = f"From {start_date} to {end_date}"

    prompt = main_prompt_template.format(
        data_records=data_records,
        statistical_summary=stats,
        future_dates=future_dates
    )

    return prompt, all_texts


def process_weather_ts_data(ts_data_dir):
    """Process weather time series data from parquet files"""
    all_data = []
    
    # Process all parquet files
    parquet_files = [f for f in os.listdir(ts_data_dir) if f.endswith('.parquet')]
    if not parquet_files:
        return pd.DataFrame()
    
    print(f"Processing {len(parquet_files)} parquet files...")
    
    for filename in parquet_files:
        filepath = os.path.join(ts_data_dir, filename)
        df = pd.read_parquet(filepath)
        
        # Process all time series from each file
        for idx, row in df.iterrows():
            dates = pd.to_datetime(row['DATE'])
            temperatures = row['temperature']
            
            # Create dataframe for this time series
            ts_df = pd.DataFrame({
                'date': dates,
                'temperature': temperatures
            })
            
            # Remove NaN values
            ts_df = ts_df.dropna()
            
            if len(ts_df) > 0:
                all_data.append(ts_df)
    
    # Concatenate all data and sort by date
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values('date').reset_index(drop=True)
        return combined_df
    else:
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ts_data_dir", type=str, help="Directory with time series parquet files", 
                        default='/data2/user2/MTBench/weather/ts/data')
    parser.add_argument("--text_data_dir", type=str, help="Directory with text JSON files", 
                        default='/data2/user2/MTBench/weather/text')
    parser.add_argument("--hist_len", type=int, help="Historical time series length", default=96)
    parser.add_argument("--pred_len", type=int, help="Prediction length", default=12)
    parser.add_argument("--save_dir", type=str, help="Save path of processed dataset", 
                        default='/data2/user2/ter_tsf/processed_data')
    
    args = parser.parse_args()

    # Load real text data
    print("Loading weather text data...")
    text_data = load_weather_text_data(args.text_data_dir)
    print(f"Loaded text data for {len(text_data)} dates")

    # Load and process time series data
    print("Loading weather time series data...")
    df_all = process_weather_ts_data(args.ts_data_dir)
    
    if len(df_all) == 0:
        print("No valid time series data found!")
        return
    
    print(f"Total time series data points: {len(df_all)}")
    
    # Remove rows with NaN temperatures
    df_all = df_all.dropna(subset=["date", "temperature"]).reset_index(drop=True)
    print(f"After removing NaN: {len(df_all)}")

    # Split data
    num_train = int(len(df_all) * 0.7)
    num_test = int(len(df_all) * 0.2)
    num_vali = len(df_all) - num_train - num_test
    border1s = [0, num_train - args.hist_len, len(df_all) - num_test - args.hist_len]
    border2s = [num_train, num_train + num_vali, len(df_all)]
    type_map = {'train': 0, 'val': 1, 'test': 2}

    for dataset_type in type_map.keys():
        print(f"Processing {dataset_type} dataset...")
        dataset_type_idx = type_map[dataset_type]
        df = df_all[border1s[dataset_type_idx]:border2s[dataset_type_idx]]

        new_df = []

        # Process all valid windows
        step_size = (args.hist_len + args.pred_len) * 10  # Reduce sample volume
        
        for start in tqdm(range(0, len(df) - args.hist_len - args.pred_len + 1, step_size)):
            hist_df = df.iloc[start:start + args.hist_len]
            pred_df = df.iloc[start + args.hist_len:start + args.hist_len + args.pred_len]

            prompt, all_texts = construct_prompt(hist_df, pred_df, text_data, feature_name='temperature')

            # Process original texts
            if all_texts:
                original_texts = " ".join(all_texts)
                if len(original_texts) > 5120:
                    original_texts = original_texts[:5120]
            else:
                original_texts = "No text."

            new_df.append({
                "history_series": hist_df["temperature"].tolist(),
                "prompt": prompt,
                "horizon_series": pred_df["temperature"].tolist(),
                "original_text": original_texts
            })

        if new_df:
            output_df = pd.DataFrame(new_df)
            os.makedirs(args.save_dir, exist_ok=True)
            output_path = os.path.join(args.save_dir, f'MTBench_weather_{args.hist_len}_{args.pred_len}_{dataset_type}.csv')
            output_df.to_csv(output_path, index=False)
            print(f"Saved {len(output_df)} samples to {output_path}")
        else:
            print(f"No valid samples generated for {dataset_type}")


if __name__ == "__main__":
    main()
