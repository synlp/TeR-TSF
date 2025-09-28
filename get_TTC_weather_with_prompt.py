import pandas as pd
import argparse
from tqdm import tqdm
import os
import re

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


def construct_prompt(window_df, future_df, feature_name="value"):
    records = []
    for _, row in window_df.iterrows():
        report = row["text"] if pd.notna(row["text"]) and row["text"].strip() != "" else "(no text)."
        record = data_record_template.format(
            date=pd.to_datetime(row["date"]),
            feature_name=feature_name,
            value=row["temp"],
            report=report
        )
        records.append(record)
    data_records = "\n".join(records)

    values = window_df["temp"]
    stats = statistical_summary_template.format(
        max_val=round(values.max(), 2),
        min_val=round(values.min(), 2),
        avg_val=round(values.mean(), 2),
        std_val=round(values.std(), 2)
    )

    # 只取第一个和最后一个日期作为预测范围
    start_date = pd.to_datetime(future_df["date"].iloc[0])
    end_date = pd.to_datetime(future_df["date"].iloc[-1])
    future_dates = f"From {start_date} to {end_date}"

    prompt = main_prompt_template.format(
        data_records=data_records,
        statistical_summary=stats,
        future_dates=future_dates
    )

    return prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file_dir", type=str, help="CSV file with Data, Value, Text columns", default='./')
    parser.add_argument("--data", type=str, help="name of domain", default='climate_2014_2023_final')
    parser.add_argument("--hist_len", type=int, help="Historical time series length", default=96)
    parser.add_argument("--pred_len", type=int, help="Prediction length", default=12)
    parser.add_argument("--save_dir", type=str, help="save path of processed dataset", default='./datasets_with_prompt')
    
    args = parser.parse_args()

    data_dir = args.csv_file_dir
    data_path = data_dir + '/' + args.data + '.csv'
    df_all = pd.read_csv(data_path)
    df_all = df_all.dropna(subset=["date", "temp"]).reset_index(drop=True)

    num_train = int(len(df_all) * 0.7)
    num_test = int(len(df_all) * 0.2)
    num_vali = len(df_all) - num_train - num_test
    border1s = [0, num_train - args.hist_len, len(df_all) - num_test - args.hist_len]
    border2s = [num_train, num_train + num_vali, len(df_all)]
    type_map = {'train': 0, 'val': 1, 'test': 2}

    for dataset_type in type_map.keys():
        dataset_type_idx = type_map[dataset_type]
        df = df_all[border1s[dataset_type_idx]:border2s[dataset_type_idx]]

        new_df = []

        for start in tqdm(range(0, len(df) - args.hist_len - args.pred_len + 1)):
            hist_df = df.iloc[start:start + args.hist_len]
            pred_df = df.iloc[start + args.hist_len:start + args.hist_len + args.pred_len]

            prompt = construct_prompt(hist_df, pred_df, feature_name='value')

            original_texts = hist_df["text"].dropna(how='any', axis=0).tolist()
            if original_texts:
                original_texts = " ".join(original_texts)
                if len(original_texts) > 5120:
                    original_texts = original_texts[:5120]
            else:
                original_texts = "No text."

            
            new_df.append({
                "history_series": hist_df["temp"].tolist(),
                "prompt": prompt,
                "horizon_series": pred_df["temp"].tolist(),
                "original_text": original_texts
            })

        output_df = pd.DataFrame(new_df)
        os.makedirs(args.save_dir, exist_ok=True)
        output_path = os.path.join(args.save_dir+'/'+f'weather_{args.hist_len}_{args.pred_len}_'+dataset_type+'.csv')
        output_df.to_csv(output_path, index=False)

        print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
