import argparse
import json
from gpu_process import GPUTaskProcess
import multiprocessing

def load_gpu_tasks(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description='Run tasks on GPUs with resuming capability.')
    parser.add_argument('--config', default='test.json', help='Path to GPU tasks configuration file.')
    parser.add_argument('--restart', action='store_true', help='Ignore previous state (not include log files) and restart all tasks.')
    args = parser.parse_args()

    exp_config = load_gpu_tasks(args.config)
    exp_name = exp_config["experiment_name"]
    exp_tasks = exp_config["tasks"]
    
    processes = []

    for gpu_id in exp_tasks.keys():
        tasks = exp_tasks[gpu_id]
        process = GPUTaskProcess(gpu_id, tasks, exp_name, restart=args.restart)
        process.start()
        processes.append(process)

    # 确保所有进程完成后再退出主进程
    for process in processes:
        process.join()

if __name__ == "__main__":
    # 标记当前模块为"已冻结"状态, 防止子进程重新执行主模块代码
    multiprocessing.freeze_support()  # 可选，用于 Windows 平台支持
    main()