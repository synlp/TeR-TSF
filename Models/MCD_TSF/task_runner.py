import argparse
import json
from gpu_worker import GPUTaskWorker

def load_gpu_tasks(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description='Run tasks on GPUs with resuming capability.')
    parser.add_argument('--config', default='gpu_tasks.json', help='Path to GPU tasks configuration file.')
    parser.add_argument('--restart', action='store_true', help='Ignore previous state and restart all tasks.')
    args = parser.parse_args()

    gpu_tasks = load_gpu_tasks(args.config)
    
    workers = []

    for gpu_id in gpu_tasks.keys():
        tasks = gpu_tasks[gpu_id]
        worker = GPUTaskWorker(gpu_id, tasks, restart=args.restart)
        workers.append(worker)
        worker.start()

    # 确保所有任务完成后再退出主线程
    for worker in workers:
        worker.join()

if __name__ == "__main__":
    main()
