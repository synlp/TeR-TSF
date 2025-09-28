import json
import os
import subprocess
import multiprocessing
import datetime

class GPUTaskProcess(multiprocessing.Process):
    def __init__(self, gpu_id, tasks, exp_name, restart=False):
        super().__init__()
        self.gpu_id = gpu_id
        self.tasks = tasks
        self.restart = restart
        self.save_dir = f"experiments/{exp_name}"
        self.state_file = os.path.join(self.save_dir, f"gpu_{gpu_id}_state.json")
        self.log_dir = os.path.join(self.save_dir, f"gpu{gpu_id}")
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
                        ["bash", task],
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    for line in process.stdout:
                        print(line, end='')
                        log_file.write(line)
                    return_code = process.wait()
                    # 检查子进程退出码
                    if return_code != 0:
                        error_msg = f"Task failed with exit code {return_code}."
                        state["status"] = "error"
                        state["error_message"] = f"Task {task} failed with exit code {return_code}. Log: {log_path}"
                        self.save_state(state)
                        print(f"GPU {self.gpu_id} task {task} failed. See {log_path}")
            except Exception as e:
                state["status"] = "error"
                state["error_message"] = f"Error executing {task}: {str(e)}"
                self.save_state(state)
                print(f"GPU {self.gpu_id} error: {e}")
                break
        state["status"] = "completed"
        state["error_message"] = None
        self.save_state(state)
        print(f"GPU {self.gpu_id} all tasks completed.")