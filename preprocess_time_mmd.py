import pandas as pd
import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, help="", default='/home/suchen/Time-MMD-main')
    parser.add_argument("--save_dir", type=str, help="", default='./timeMMD_processed')
    args = parser.parse_args()

    domain_ls = ['Agriculture', 'Climate', 'Economy', 'Energy', 'Environment', 'Health_US', 'SocialGood', 'Traffic']

    for domain in domain_ls:
        # 读取 CSV 文件
        df_ts = pd.read_csv(os.path.join(args.dataset_dir, f'numerical/{domain}/{domain}.csv'))
        df_text = pd.read_csv(os.path.join(args.dataset_dir, f'textual/{domain}/{domain}_report.csv'))

        df_ts = df_ts[['start_date', 'OT']]
        df_text = df_text[['start_date', 'fact']]

        # 确保日期列为日期格式
        df_ts['start_date'] = pd.to_datetime(df_ts['start_date'])
        df_text['start_date'] = pd.to_datetime(df_text['start_date'])
        
        # 合并相同日期的文本
        df_text = df_text.groupby('start_date').agg(lambda x: ' '.join(x.dropna().astype(str)))

        # 合并两个数据框，按 end_date 匹配
        merged_df = pd.merge(df_ts, df_text, on=['start_date'], how='left', sort=True)

        # 改列名
        merged_df = merged_df.rename(columns={'start_date': 'date', 'OT': 'value', 'fact': 'text'})

        os.makedirs(args.save_dir, exist_ok=True)

        # 保存结果到新的 CSV 文件
        merged_df.to_csv(os.path.join(args.save_dir, f'{domain}.csv'))

        print(f"Raw dataset length: {len(df_ts)} -> processed dataset length: {len(merged_df)}")

if __name__ == "__main__":
    main()