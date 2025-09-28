import pandas as pd
import argparse
import os
import json

def main():
    parser = argparse.ArgumentParser(description="LLM Prediction with Examples")
    parser.add_argument("--csv_dir", type=str, help="Path to the CSV file", default="/media/ubuntu/data/collaborations/tsf/TeR-TSF/reinforced_my_datasets/Time-MMD/Traffic")
    parser.add_argument("--data", type=str, help="data", default="Traffic")
    parser.add_argument("--hist_len", type=int, help="Historical time series length", default=36)
    parser.add_argument("--chosen_num", type=int, help="", default=2)
    parser.add_argument("--pred_len", type=int, help="Prediction length", default=6)
    parser.add_argument("--save_dir", type=str, help="dir to save dataset", default='/media/ubuntu/data/collaborations/tsf/TeR-TSF/preference_datasets')
    args = parser.parse_args()

    df1 = pd.read_csv(os.path.join(args.csv_dir, f'{args.data}_{args.hist_len}_{args.pred_len}_train_0.csv'))
    df1['reward_0'] = df1.iloc[:, 4:].sum(axis=1)
    df1 = df1[['history_series', 'horizon_series', 'prompt', 'reinforced_text', 'reward_0']]
    df1.rename(columns={'reinforced_text': 'reinforced_text_0'}, inplace=True)
    for idx in range(1, args.chosen_num):
        df2 = pd.read_csv(os.path.join(args.csv_dir, f'{args.data}_{args.hist_len}_{args.pred_len}_train_{idx}.csv'))
        df2[f'reward_{idx}'] = df2.iloc[:, 4:].sum(axis=1)
        df2 = df2[['history_series', 'reinforced_text', f'reward_{idx}']]
        df2.rename(columns={'reinforced_text': f'reinforced_text_{idx}'}, inplace=True)
        df1 = pd.merge(df1, df2, how='inner', on='history_series')
    
    select_cols_name = [c for c in df1.columns if 'reward' in c]
    reward_cols = df1[select_cols_name]
    pos = reward_cols.idxmax(axis=1).apply(lambda x: 'reinforced_text'+x[-2:])
    neg = reward_cols.idxmin(axis=1).apply(lambda x: 'reinforced_text'+x[-2:])
    preference_dataset = []
    for iii in range(len(pos)):
        preference_dataset.append({
    "conversations": [
      {
        "from": "human",
        "value": df1.loc[iii, 'prompt']
      }
    ],
    "chosen": {
      "from": "gpt",
      "value": f"<summary>{df1.loc[iii, pos[iii]]}</summary>"
    },
    "rejected": {
      "from": "gpt",
      "value": f"<summary>{df1.loc[iii, neg[iii]]}</summary>"
    }
  })
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f'{args.data}_h{args.hist_len}_p{args.pred_len}_n{args.chosen_num}.json')
    with open(save_path, 'w') as file_obj:
        json.dump(preference_dataset, file_obj, indent=4)

    # 把生成的偏好数据集信息更新到LLaMA-Factory-main
    new_data_info = {
        f"{args.data}_h{args.hist_len}_p{args.pred_len}_n{args.chosen_num}": {
        "file_name": save_path,
        "ranking": True,
        "formatting": "sharegpt", 
        "columns": {
          "messages": "conversations",
          "chosen": "chosen",
          "rejected": "rejected"}
        }
    }
    with open("./LLaMA-Factory-main/data/dataset_info.json", "r", encoding="utf-8") as f:
        data_info = json.load(f)
        data_info.update(new_data_info)
    with open("./LLaMA-Factory-main/data/dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(data_info, f, indent=4)

if __name__ == "__main__":
    main()