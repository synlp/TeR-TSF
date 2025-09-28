"""
Calculate mean and std for MTBench datasets (weather and finance)
Similar to norm_stat.py for other datasets
"""

from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os

def calculate_weather_stats():
    """Calculate mean and std for MTBench weather dataset"""
    print("=== MTBench Weather Dataset ===")
    
    # Load weather time series data
    ts_data_dir = '/data2/user2/MTBench/weather/ts/data'
    parquet_files = [f for f in os.listdir(ts_data_dir) if f.endswith('.parquet')]
    
    if not parquet_files:
        print("No weather parquet files found!")
        return
    
    print(f"Found {len(parquet_files)} weather parquet files")
    
    all_data = []
    
    # Process all parquet files
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
        print(f"Total weather data points: {len(combined_df)}")
        
        # Split data (70% train, 20% test, 10% val)
        hist_len = 96
        num_train = int(len(combined_df) * 0.7)
        num_test = int(len(combined_df) * 0.2)
        num_vali = len(combined_df) - num_train - num_test
        border1s = [0, num_train - hist_len, len(combined_df) - num_test - hist_len]
        border2s = [num_train, num_train + num_vali, len(combined_df)]
        
        # Get temperature data for training
        df_data = combined_df[['temperature']]
        train_data = df_data[border1s[0]:border2s[0]]
        
        # Calculate stats
        scaler = StandardScaler()
        scaler.fit(train_data.values)
        mean_data = scaler.mean_
        std_data = scaler.scale_
        
        print(f"Weather dataset - mean: {mean_data}, std: {std_data}")
        return mean_data, std_data
    else:
        print("No valid weather data found!")
        return None, None

def calculate_finance_stats():
    """Calculate mean and std for MTBench finance dataset"""
    print("\n=== MTBench Finance Dataset ===")
    
    # Load finance time series data
    ts_data_dir = '/data2/user2/MTBench/finance/ts'
    parquet_files = [f for f in os.listdir(ts_data_dir) if f.endswith('.parquet')]
    parquet_files.sort()
    
    if not parquet_files:
        print("No finance parquet files found!")
        return
    
    print(f"Found {len(parquet_files)} finance parquet files")
    
    all_data = []
    
    # Process all parquet files
    for filename in parquet_files:
        filepath = os.path.join(ts_data_dir, filename)
        df = pd.read_parquet(filepath)
        
        # Process all rows from each file
        for idx, row in df.iterrows():
            timestamps = row['timestamp']
            closes = row['close']
            
            # Create dataframe for this time series
            ts_df = pd.DataFrame({
                'timestamp': timestamps,
                'close': closes
            })
            
            # Remove NaN values
            ts_df = ts_df.dropna()
            
            if len(ts_df) > 0:
                all_data.append(ts_df)
    
    # Concatenate all data and sort by timestamp
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        
        # Remove duplicate timestamps
        combined_df = combined_df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        print(f"Total finance data points: {len(combined_df)}")
        
        # Split data (70% train, 20% test, 10% val)
        hist_len = 96
        num_train = int(len(combined_df) * 0.7)
        num_test = int(len(combined_df) * 0.2)
        num_vali = len(combined_df) - num_train - num_test
        border1s = [0, num_train - hist_len, len(combined_df) - num_test - hist_len]
        border2s = [num_train, num_train + num_vali, len(combined_df)]
        
        # Get close price data for training
        df_data = combined_df[['close']]
        train_data = df_data[border1s[0]:border2s[0]]
        
        # Calculate stats
        scaler = StandardScaler()
        scaler.fit(train_data.values)
        mean_data = scaler.mean_
        std_data = scaler.scale_
        
        print(f"Finance dataset - mean: {mean_data}, std: {std_data}")
        return mean_data, std_data
    else:
        print("No valid finance data found!")
        return None, None

def test_finance_normalization():
    """Test the finance normalization logic with a small sample"""
    print("\n=== Testing Finance Normalization ===")
    
    from sklearn.preprocessing import StandardScaler
    
    # Load finance time series data
    ts_data_dir = '/data2/user2/MTBench/finance/ts'
    parquet_files = [f for f in os.listdir(ts_data_dir) if f.endswith('.parquet')]
    parquet_files.sort()
    
    if not parquet_files:
        print("No finance parquet files found!")
        return
    
    # Test with first file only
    filename = parquet_files[0]
    filepath = os.path.join(ts_data_dir, filename)
    df = pd.read_parquet(filepath)
    
    print(f"Testing with {filename}: {len(df)} stocks")
    
    # Process first 3 stocks as test
    for idx in range(min(3, len(df))):
        row = df.iloc[idx]
        timestamps = row['timestamp']
        closes = row['close']
        
        # Create dataframe for this stock
        stock_df = pd.DataFrame({
            'timestamp': timestamps,
            'close': closes
        })
        
        # Remove NaN values
        stock_df = stock_df.dropna()
        
        if len(stock_df) > 100:
            print(f"\nStock {idx}:")
            print(f"  Original - Mean: {stock_df['close'].mean():.2f}, Std: {stock_df['close'].std():.2f}")
            print(f"  Original - Min: {stock_df['close'].min():.2f}, Max: {stock_df['close'].max():.2f}")
            
            # Normalize this stock individually
            scaler = StandardScaler()
            normalized_closes = scaler.fit_transform(stock_df[['close']])
            
            print(f"  Normalized - Mean: {normalized_closes.mean():.6f}, Std: {normalized_closes.std():.6f}")
            print(f"  Normalized - Min: {normalized_closes.min():.2f}, Max: {normalized_closes.max():.2f}")


if __name__ == "__main__":
    # Calculate stats for both datasets
    weather_mean, weather_std = calculate_weather_stats()
    finance_mean, finance_std = calculate_finance_stats()
    
    print("\n=== Summary ===")
    if weather_mean is not None:
        print(f"Weather - mean: {weather_mean[0]:.6f}, std: {weather_std[0]:.6f}")
    if finance_mean is not None:
        print(f"Finance - mean: {finance_mean[0]:.6f}, std: {finance_std[0]:.6f}")
    
    # Test normalization logic
    test_finance_normalization()
