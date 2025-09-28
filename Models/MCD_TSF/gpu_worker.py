import json
import os
import subprocess
import threading
import datetime
class GPUTaskWorker(threading.Thread):
    def __init__(self, gpu_id, tasks, restart=False):
        super().__init__()
        self.gpu_id = gpu_id
        self.tasks = tasks
        self.restart = restart
        self.state_dir = "states"
        self.state_file = os.path.join(self.state_dir, f"gpu_{gpu_id}_state.json")
        self.log_dir = os.path.join("logs", f"gpu{gpu_id}")
        os.makedirs(self.state_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def load_state(self):
        if self.restart or not os.path.exists(self.state_file):
            return None
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                if 'current_index' in state and 'tasks' in state:
                    return state
                else:
                    print(f"Invalid state file for GPU {self.gpu_id}, resetting.")
                    return None
        except Exception as e:
            print(f"Error loading state for GPU {self.gpu_id}: {e}, resetting.")
            return None

    def save_state(self, state):
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def run(self):
        state = self.load_state()
        if state is None:
            state = {
                "gpu_id": self.gpu_id,
                "current_index": 0,
                "status": "running",
                "tasks": self.tasks,
                "error_message": None
            }
        else:
            if state.get("status") == "completed":
                print(f"GPU {self.gpu_id} tasks already completed.")
                return

        current_index = state["current_index"]
        tasks = self.tasks

        for idx in range(current_index, len(tasks)):
            task = tasks[idx]
            log_path = os.path.join(self.log_dir, f"task_{idx}.log")
            state["current_index"] = idx
            state["status"] = "running"
            state["error_message"] = None
            self.save_state(state)

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

            try:
                with open(log_path, 'a') as log_file:
                    log_file.write(f"{datetime.datetime.now()} 执行新的代码：\n")
                    process = subprocess.Popen(
                        ["bash", task], # 要执行的命令
                        env=env, # 执行环境
                        stdout=subprocess.PIPE, # 处理正常输出
                        stderr=subprocess.STDOUT, # 处理错误输出，这里合并到stdout
                        text=True, # 以文本模式打开（即字符串形式）。默认情况下，以二进制模式打开（即字节序列）。
                    )
                    for line in process.stdout:
                        print(line, end='')
                        # 将输出同时写入日志文件
                        log_file.write(line)
                    process.wait()
                state["current_index"] += 1
            except subprocess.CalledProcessError as e:
                state["status"] = "error"
                state["error_message"] = f"Task {task} failed with exit code {e.returncode}. Log: {log_path}"
                self.save_state(state)
                print(f"GPU {self.gpu_id} task {task} failed. See {log_path}")
                break
            except Exception as e:
                state["status"] = "error"
                state["error_message"] = f"Error executing {task}: {str(e)}"
                self.save_state(state)
                print(f"GPU {self.gpu_id} error: {e}")
                break
        # else:
        #     state["status"] = "completed"
        #     state["error_message"] = None
        #     self.save_state(state)
        #     print(f"GPU {self.gpu_id} all tasks completed.")
