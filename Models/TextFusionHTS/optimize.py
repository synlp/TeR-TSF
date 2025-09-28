import subprocess
import optuna
from optuna.samplers import TPESampler
import argparse
import sys

sys.stdout.reconfigure(line_buffering=True) # 全局行缓冲

# 自定义全局早停回调
import optuna

class GlobalStoppingCallback:
    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self._best_value = None
        self._counter = 0
        
    def __call__(self, study, trial):
        current_value = study.best_value
        
        # 初始化最佳值（仅第一次运行时执行）
        if self._best_value is None:
            self._best_value = current_value
            return
        
        # 检查是否处于TPESampler的启动阶段
        in_startup_phase = False
        if isinstance(study.sampler, optuna.samplers.TPESampler):
            completed_trials = len(study.trials)  # 已完成试验总数
            in_startup_phase = completed_trials < study.sampler._n_startup_trials
        
        # 在启动阶段跳过早停检查
        if in_startup_phase:
            # 启动阶段仍更新最佳值，但不更新计数器
            if current_value < self._best_value - self.min_delta:
                self._best_value = current_value
            return
        
        # 主逻辑：检查提升是否显著
        if current_value < self._best_value - self.min_delta:
            self._best_value = current_value
            self._counter = 0  # 重置计数器
        else:
            self._counter += 1  # 提升不足时增加计数器
        
        # 检查早停条件
        if self._counter >= self.patience:
            study.stop()  # 触发早停

def objective(trial):
    # 定义超参数空间
    params = {
        "--lr": trial.suggest_categorical("lr", [0.00001, 0.000025, 0.00005, 0.000075, 0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0025, 0.005, 0.0075, 0.01]),
        "--batch_size": trial.suggest_categorical("batch_size", [8, 16, 32, 64]),
        "--dropout": trial.suggest_categorical("dropout", [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
    }

    # 固定参数task
    fixed_params = [
        "--data_dir", args.data_dir,
        "--save_dir", args.save_dir,
        "--data_name", args.data_name,
        "--hist_len", args.hist_len,
        "--pred_len", args.pred_len,
        "--text_type", args.text_type,
        "--exp_time", args.exp_time,
        "--iter_idx", args.iter_idx,
        "--epochs", args.epochs
    ]
    
    # 构造命令行命令
    # cmd = ["./your_script.sh"] + fixed_params # 对于shell脚本的情况
    cmd = ["python", "Models/TextFusionHTS/train_tfhts.py"] + fixed_params
    for key, value in params.items():
        cmd += [key, str(value)]
    
    
    # 执行实验
    proc = subprocess.Popen(cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True, # 如果为 True，则 stdin、stdout 和 stderr 将以文本模式打开（即字符串形式）。默认情况下，这些流以二进制模式打开（即字节序列）。
                            bufsize=1) # 行缓冲模式, 确保实时输出
    
    last_line = ""
    for line in proc.stdout:
        # 实时打印输出（保留原始格式）
        sys.stdout.write(line)
        sys.stdout.flush()
        last_line = line.strip()

    # 等待进程结束并获取返回码
    return_code = proc.wait()
    
    # 检查进程是否成功完成
    if return_code != 0:
        raise RuntimeError(f"Training process failed with exit code {return_code}")
    
    # 尝试从最后一行提取数值结果
    try:
        return float(last_line)
    except ValueError:
        raise ValueError(f"Could not convert training output to float: '{last_line}'. Please ensure the training script outputs a numeric value on the last line.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TeR-TSF-Study")
    parser.add_argument("--study_name", type=str, default="")
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--data_name", type=str, default="")
    parser.add_argument("--hist_len", type=str, default="")
    parser.add_argument("--pred_len", type=str, default="")
    parser.add_argument("--text_type", type=str, default="")
    parser.add_argument("--exp_time", type=str, default="")
    parser.add_argument("--iter_idx", type=str, default="")
    parser.add_argument("--epochs", type=str, default="")
    args = parser.parse_args()
    # print(args)

    tep_sampler= TPESampler(
        n_startup_trials=25,             # 初始随机探索次数，提供较好的初始覆盖。
        n_ei_candidates=25,             # 候选点数量（提高精度）
        # gamma=lambda n: 0.5 * n // 1,  # 更严格的好样本比例，加速向有希望区域的收敛。
        multivariate=True,              # 启用多变量模式（考虑参数相关性）
        group=False,                    # 处理条件参数依赖
        prior_weight=1.2,               # 增强先验分布的影响
        weights=lambda n: [max(1e-5, 0.98**(n-i)) for i in range(n)], # 慢衰减权重
        seed=42,
        consider_magic_clip=True,       # 启用自动边界裁剪
        consider_endpoints=True        # 关注边界点
    )
    
    study = optuna.create_study(direction="minimize",
                                sampler=tep_sampler,
                                study_name=args.study_name)
    
    study.optimize(objective,   # 目标函数
                   n_trials=100, # 试验次数
                   timeout=None,# 优化时间限制（秒）
                   n_jobs=1,    # 并行任务数
                   callbacks=[GlobalStoppingCallback(patience=5, min_delta=0.001)], # 早停
                   show_progress_bar=True # 是否显示进度条
                   )
    
    # 获取最优的前 k 个试验
    k = 5  # 设置需要提取的 top k 数量
    # 步骤 1: 获取所有完成的试验（过滤掉失败/未完成的）
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    # 步骤 2: 按目标值排序（考虑优化方向）
    if study.direction == optuna.study.StudyDirection.MAXIMIZE:
        sorted_trials = sorted(complete_trials, key=lambda t: t.value, reverse=True)
    else:
        sorted_trials = sorted(complete_trials, key=lambda t: t.value)
    # 步骤 3: 取前 k 个试验
    top_k_trials = sorted_trials[:k]
    # 打印结果
    print(f"Top {k} trials:")
    for i, trial in enumerate(top_k_trials):
        print(f"\nRank {i+1}:")
        print(f"  Value: {trial.value}")
        print(f"  Params: {trial.params}")
        print(f"  Trial ID: {trial.number}")
