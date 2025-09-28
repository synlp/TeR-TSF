import subprocess
import optuna
import argparse

def objective(trial):
    # 定义超参数空间
    params = {
        "--epochs": trial.suggest_int("epochs", 50, 150, step=25),
        "--lr": trial.suggest_float("lr", 1e-4, 1e-2, step=None),
        "--beta_end": trial.suggest_float("beta_end", 0.3, 0.7, step=0.1),
        "--c_mask_prob": trial.suggest_float("c_mask_prob", 0.05, 0.3, step=0.05),
        "--num_steps": trial.suggest_int("num_steps", 100, 250, step=25),
        "--sample_steps": trial.suggest_int("sample_steps", 25, 105, step=10),
    }

    # 固定参数task
    fixed_params = [
        "--root_path", args.root_path,
        "--config", args.config,
        "--data", "pretrain",
        "--data_name", args.data_name,
        "--seq_len", args.seq_len,
        "--pred_len", args.pred_len,
        "--freq", args.freq,
        # "--epochs", "1"
    ]
    
    # 构造命令行命令
    cmd = ["python", "exe_forecasting.py"] + fixed_params
    for key, value in params.items():
        cmd += [key, str(value)]
    
    # 执行实验
    proc = subprocess.Popen(cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True # 如果为 True，则 stdin、stdout 和 stderr 将以文本模式打开（即字符串形式）。默认情况下，这些流以二进制模式打开（即字节序列）。
                            )
    for line in proc.stdout:
        print(line, end='')

    # 等待进程结束并获取返回码
    return_code = proc.wait()
    # if return_code != 0:
    #     raise RuntimeError(f"Command failed with exit code {return_code}")

    return float(line.strip())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCD-TSF-Study")
    parser.add_argument("--study_name", type=str, default="Energy_96_12")
    parser.add_argument("--root_path", type=str, default="/media/ubuntu/data/collaborations/tsf/TeR-TSF/my_datasets/Time-MMD")
    parser.add_argument("--config", type=str, default="config/energy_96_12.yaml")
    parser.add_argument("--data_name", type=str, default="Energy")
    parser.add_argument("--seq_len", type=str, default="96")
    parser.add_argument("--pred_len", type=str, default="12")
    parser.add_argument("--freq", type=str, default="w")
    args = parser.parse_args()
    print(args)
    
    study = optuna.create_study(direction="minimize",
                                study_name=args.study_name)
    
    study.optimize(objective,   # 目标函数
                   n_trials=10, # 试验次数
                   timeout=None,# 优化时间限制（秒）
                   n_jobs=1,    # 并行任务数
                   show_progress_bar=True # 是否显示进度条
                   )
    
    # 输出最佳结果
    print(f"Best {args.study_name} trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
