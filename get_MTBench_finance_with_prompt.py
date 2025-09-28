"""
MTBench Finance Dataset Processing Script

This script processes finance data from MTBench and converts it into the format required
for time series forecasting with text prompts. It generates CSV files with the following columns:
- history_series: Historical time series data (normalized close prices)
- prompt: Formatted prompt with historical data and statistics
- horizon_series: Future prediction targets
- original_text: Associated text information

The script features:
1. Processing all available finance time series data
2. Using real finance news text data matched by timestamp
3. Individual stock normalization to handle different price ranges
4. Generating complete datasets without sampling limitations

Usage:
    python get_MTBench_finance_with_prompt.py --hist_len 96 --pred_len 12
    python get_MTBench_finance_with_prompt.py --hist_len 96 --pred_len 24
    python get_MTBench_finance_with_prompt.py --hist_len 96 --pred_len 48
"""

import pandas as pd
import argparse
from tqdm import tqdm
import os
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

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


def load_finance_text_data(text_data_path):
    """Load finance text data from parquet file"""
    df = pd.read_parquet(text_data_path)
    
    # Convert published_utc to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['published_utc']):
        df['published_utc'] = pd.to_datetime(df['published_utc'])
    
    # Create a mapping from date to news content
    text_data = {}
    for _, row in df.iterrows():
        # Convert to date for daily matching
        date_key = row['published_utc'].date()
        if date_key not in text_data:
            text_data[date_key] = []
        
        # Combine title and content
        title = row['title'] if pd.notna(row['title']) else ""
        content = row['content'] if pd.notna(row['content']) else ""
        description = row['description'] if pd.notna(row['description']) else ""
        
        # Create full text entry
        full_text = f"{title}. {description}. {content}".strip()
        if full_text and full_text != "..":
            text_data[date_key].append(full_text)
    
    return text_data


def find_relevant_text(timestamp_ms, text_data):
    """Find relevant news text for a given timestamp (only historical texts)"""
    # Convert timestamp from milliseconds to datetime
    target_date = datetime.fromtimestamp(timestamp_ms / 1000).date()
    
    relevant_texts = []
    
    # Look for texts on the same date or previous dates only (no future data)
    for delta_days in range(-2, 1):  # Only -2, -1, 0 days (no future)
        check_date = target_date + timedelta(days=delta_days)
        if check_date in text_data:
            relevant_texts.extend(text_data[check_date])
    
    return relevant_texts


def construct_prompt(window_df, future_df, text_data, feature_name="close"):
    records = []
    all_texts = []
    
    for _, row in window_df.iterrows():
        timestamp = row["timestamp"]
        value = row[feature_name]
        
        # Convert timestamp to readable date
        readable_date = datetime.fromtimestamp(timestamp / 1000)
        
        # Find relevant text for this timestamp
        relevant_texts = find_relevant_text(timestamp, text_data)
        if relevant_texts:
            # Use first relevant text, limit length
            report = relevant_texts[0][:300]  # Limit to 300 characters
            all_texts.extend(relevant_texts)
        else:
            report = "(no text)."
        
        record = data_record_template.format(
            date=readable_date,
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
    start_timestamp = future_df["timestamp"].iloc[0]
    end_timestamp = future_df["timestamp"].iloc[-1]
    start_date = datetime.fromtimestamp(start_timestamp / 1000)
    end_date = datetime.fromtimestamp(end_timestamp / 1000)
    future_dates = f"From {start_date} to {end_date}"

    prompt = main_prompt_template.format(
        data_records=data_records,
        statistical_summary=stats,
        future_dates=future_dates
    )

    return prompt, all_texts


def process_finance_ts_data_with_normalization(ts_data_dir):
    """Process finance time series data with individual stock normalization"""
    all_data = []
    
    # Get all parquet files
    parquet_files = [f for f in os.listdir(ts_data_dir) if f.endswith('.parquet')]
    parquet_files.sort()
    
    if not parquet_files:
        return pd.DataFrame()
    
    print(f"Found {len(parquet_files)} parquet files, processing with individual stock normalization...")
    
    stock_id = 0
    
    for filename in parquet_files:
        filepath = os.path.join(ts_data_dir, filename)
        df = pd.read_parquet(filepath)
        
        print(f"Processing {filename}: {len(df)} stocks...")
        
        # Process each row (each stock) separately
        for idx, row in df.iterrows():
            timestamps = row['timestamp']
            closes = row['close']
            
            # Create dataframe for this stock
            stock_df = pd.DataFrame({
                'timestamp': timestamps,
                'close': closes,
                'stock_id': stock_id
            })
            
            # Remove NaN values
            stock_df = stock_df.dropna()
            
            # if len(stock_df) > 100:  # Only process stocks with sufficient data
            # Normalize this stock individually
            scaler = StandardScaler()
            stock_df['close_normalized'] = scaler.fit_transform(stock_df[['close']])
            
            # Replace original close with normalized values
            stock_df['close'] = stock_df['close_normalized']
            stock_df = stock_df.drop('close_normalized', axis=1)
            
            all_data.append(stock_df)
            
            stock_id += 1
    
    # Concatenate all normalized data and sort by timestamp
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        
        # Remove duplicate timestamps (keep first occurrence)
        combined_df = combined_df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        
        print(f"Total normalized data points: {len(combined_df)}")
        print(f"Processed {stock_id} individual stocks")
        
        return combined_df
    else:
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ts_data_dir", type=str, help="Directory with time series parquet files", 
                        default='/data2/user2/MTBench/finance/ts')
    parser.add_argument("--text_data_path", type=str, help="Path to text data parquet file", 
                        default='/data2/user2/MTBench/finance/text/data/train-00000-of-00001.parquet')
    parser.add_argument("--hist_len", type=int, help="Historical time series length", default=96)
    parser.add_argument("--pred_len", type=int, help="Prediction length", default=96)
    parser.add_argument("--save_dir", type=str, help="Save path of processed dataset", 
                        default='/data2/user2/ter_tsf/processed_data')
    
    args = parser.parse_args()

    # Load real text data
    print("Loading finance text data...")
    text_data = load_finance_text_data(args.text_data_path)
    print(f"Loaded text data for {len(text_data)} dates")

    # Load and process time series data with individual stock normalization
    print("Loading finance time series data with individual stock normalization...")
    df_all = process_finance_ts_data_with_normalization(args.ts_data_dir)
    
    if len(df_all) == 0:
        print("No valid time series data found!")
        return
    
    print(f"Total normalized time series data points: {len(df_all)}")
    
    # Remove rows with NaN close prices
    df_all = df_all.dropna(subset=["timestamp", "close"]).reset_index(drop=True)
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
        step_size = args.hist_len + args.pred_len  # No sampling, process all windows
        
        for start in tqdm(range(0, len(df) - args.hist_len - args.pred_len + 1, step_size)):
            hist_df = df.iloc[start:start + args.hist_len]
            pred_df = df.iloc[start + args.hist_len:start + args.hist_len + args.pred_len]

            prompt, all_texts = construct_prompt(hist_df, pred_df, text_data, feature_name='close')

            # Process original texts
            if all_texts:
                original_texts = " ".join(all_texts)
                if len(original_texts) > 5120:
                    original_texts = original_texts[:5120]
            else:
                original_texts = "No text."

            new_df.append({
                "history_series": hist_df["close"].tolist(),
                "prompt": prompt,
                "horizon_series": pred_df["close"].tolist(),
                "original_text": original_texts
            })

        if new_df:
            output_df = pd.DataFrame(new_df)
            os.makedirs(args.save_dir, exist_ok=True)
            output_path = os.path.join(args.save_dir, f'MTBench_finance_{args.hist_len}_{args.pred_len}_{dataset_type}.csv')
            output_df.to_csv(output_path, index=False)
            print(f"Saved {len(output_df)} samples to {output_path}")
        else:
            print(f"No valid samples generated for {dataset_type}")


if __name__ == "__main__":
    main()
