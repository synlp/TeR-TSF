"""
TeR-TSF 实时监测工具
用于监控长时间运行的实验进程，防止卡死并提供实时状态信息
"""

import time
import psutil
import threading
import subprocess
import os
import signal
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import GPUtil
from collections import deque


class ProcessMonitor:
    """进程实时监控器"""
    
    def __init__(self, process_name: str, log_file: str, output_dir: str = "./monitoring"):
        self.process_name = process_name
        self.log_file = log_file
        self.output_dir = output_dir
        self.monitoring = False
        self.monitor_thread = None
        self.process = None
        self.start_time = None
        
        # 监控数据
        self.cpu_history = deque(maxlen=60)  # 保存最近60次CPU使用率
        self.memory_history = deque(maxlen=60)  # 保存最近60次内存使用
        self.gpu_history = deque(maxlen=60)  # 保存最近60次GPU使用
        self.log_size_history = deque(maxlen=60)  # 保存最近60次日志文件大小
        
        # 卡死检测参数
        self.stall_threshold = 300  # 5分钟无日志更新视为可能卡死
        self.last_log_size = 0
        self.last_log_update = time.time()
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"进程监控器已初始化: {process_name}")
        print(f"日志文件: {log_file}")
        print(f"监控输出: {output_dir}")
    
    def start_monitoring(self, process: subprocess.Popen):
        """开始监控指定进程"""
        self.process = process
        self.start_time = time.time()
        self.monitoring = True
        
        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        print(f"开始监控进程 {self.process_name} (PID: {process.pid})")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        # 生成最终报告
        self._generate_final_report()
        print(f"停止监控进程 {self.process_name}")
    
    def _monitor_loop(self):
        """监控主循环"""
        while self.monitoring and self.process and self.process.poll() is None:
            try:
                # 收集系统资源信息
                self._collect_system_metrics()
                
                # 检查日志文件更新
                self._check_log_progress()
                
                # 检测是否卡死
                self._detect_stall()
                
                # 输出实时状态
                self._print_status()
                
                # 保存监控数据
                self._save_monitoring_data()
                
                time.sleep(5)  # 每5秒监控一次
                
            except Exception as e:
                print(f"监控线程错误: {e}")
                time.sleep(5)
    
    def _collect_system_metrics(self):
        """收集系统指标"""
        try:
            # CPU和内存
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            self.cpu_history.append(cpu_percent)
            self.memory_history.append(memory.percent)
            
            # GPU信息
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_load = gpus[0].load * 100  # 使用第一个GPU
                    gpu_memory = gpus[0].memoryUtil * 100
                    self.gpu_history.append({'load': gpu_load, 'memory': gpu_memory})
                else:
                    self.gpu_history.append({'load': 0, 'memory': 0})
            except:
                self.gpu_history.append({'load': 0, 'memory': 0})
                
        except Exception as e:
            print(f"收集系统指标失败: {e}")
    
    def _check_log_progress(self):
        """检查日志文件进度"""
        try:
            if os.path.exists(self.log_file):
                current_size = os.path.getsize(self.log_file)
                
                if current_size > self.last_log_size:
                    self.last_log_update = time.time()
                    self.last_log_size = current_size
                
                self.log_size_history.append(current_size)
            else:
                self.log_size_history.append(0)
                
        except Exception as e:
            print(f"检查日志进度失败: {e}")
    
    def _detect_stall(self):
        """检测进程是否卡死（静默模式）"""
        current_time = time.time()
        time_since_log_update = current_time - self.last_log_update
        
        if time_since_log_update > self.stall_threshold:
            # 静默检测，不输出警告信息
            # print(f"\n⚠️  警告：进程可能卡死！")
            # print(f"   已 {time_since_log_update/60:.1f} 分钟无日志更新")
            # print(f"   建议检查进程状态或考虑重启")
            
            # 检查进程是否还在运行
            if self.process and self.process.poll() is None:
                try:
                    # 检查进程CPU使用率
                    proc = psutil.Process(self.process.pid)
                    cpu_percent = proc.cpu_percent()
                    
                    if cpu_percent < 1.0:  # CPU使用率很低
                        # print(f"   进程CPU使用率很低: {cpu_percent:.1f}%")
                        # print(f"   可能已经卡死，考虑发送SIGTERM信号")
                        pass
                except:
                    pass
    
    def _print_status(self):
        """输出实时状态（静默模式）"""
        if not self.cpu_history:
            return
            
        current_time = time.time()
        elapsed_time = current_time - self.start_time if self.start_time else 0
        
        # 计算平均值
        avg_cpu = sum(self.cpu_history) / len(self.cpu_history)
        avg_memory = sum(self.memory_history) / len(self.memory_history)
        
        gpu_info = ""
        if self.gpu_history and self.gpu_history[-1]['load'] > 0:
            recent_gpu = list(self.gpu_history)[-10:]  # 最近10次
            avg_gpu_load = sum(g['load'] for g in recent_gpu) / len(recent_gpu)
            avg_gpu_memory = sum(g['memory'] for g in recent_gpu) / len(recent_gpu)
            gpu_info = f" | GPU: {avg_gpu_load:.1f}%负载, {avg_gpu_memory:.1f}%内存"
        
        # 日志文件大小
        log_size_mb = self.log_size_history[-1] / 1024 / 1024 if self.log_size_history else 0
        time_since_update = (current_time - self.last_log_update) / 60
        
        # 注释掉实时状态输出
        # print(f"\r🔄 [{self._format_time(elapsed_time)}] "
        #       f"CPU: {avg_cpu:.1f}% | 内存: {avg_memory:.1f}%{gpu_info} | "
        #       f"日志: {log_size_mb:.1f}MB ({time_since_update:.1f}min前更新)", 
        #       end="", flush=True)
    
    def _save_monitoring_data(self):
        """保存监控数据"""
        try:
            timestamp = datetime.now().isoformat()
            
            monitoring_data = {
                'timestamp': timestamp,
                'process_name': self.process_name,
                'elapsed_time': time.time() - self.start_time if self.start_time else 0,
                'cpu_usage': list(self.cpu_history),
                'memory_usage': list(self.memory_history),
                'gpu_usage': list(self.gpu_history),
                'log_size_history': list(self.log_size_history),
                'last_log_update': self.last_log_update,
                'time_since_log_update': time.time() - self.last_log_update,
            }
            
            # 保存到文件
            monitor_file = os.path.join(self.output_dir, f"{self.process_name}_monitoring.json")
            with open(monitor_file, 'w') as f:
                json.dump(monitoring_data, f, indent=2)
                
        except Exception as e:
            print(f"\n保存监控数据失败: {e}")
    
    def _generate_final_report(self):
        """生成最终监控报告"""
        if not self.start_time:
            return
            
        total_time = time.time() - self.start_time
        
        report = {
            'process_name': self.process_name,
            'total_runtime': total_time,
            'total_runtime_formatted': self._format_time(total_time),
            'final_log_size_mb': self.log_size_history[-1] / 1024 / 1024 if self.log_size_history else 0,
            'avg_cpu_usage': sum(self.cpu_history) / len(self.cpu_history) if self.cpu_history else 0,
            'avg_memory_usage': sum(self.memory_history) / len(self.memory_history) if self.memory_history else 0,
            'max_memory_usage': max(self.memory_history) if self.memory_history else 0,
            'stall_detected': (time.time() - self.last_log_update) > self.stall_threshold,
            'final_timestamp': datetime.now().isoformat(),
        }
        
        # GPU统计
        if self.gpu_history and any(g['load'] > 0 for g in self.gpu_history):
            gpu_loads = [g['load'] for g in self.gpu_history if g['load'] > 0]
            gpu_memories = [g['memory'] for g in self.gpu_history if g['memory'] > 0]
            
            if gpu_loads:
                report['avg_gpu_load'] = sum(gpu_loads) / len(gpu_loads)
                report['max_gpu_load'] = max(gpu_loads)
            
            if gpu_memories:
                report['avg_gpu_memory'] = sum(gpu_memories) / len(gpu_memories)
                report['max_gpu_memory'] = max(gpu_memories)
        
        # 保存报告
        report_file = os.path.join(self.output_dir, f"{self.process_name}_final_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 最终监控报告已保存: {report_file}")
        print(f"   总运行时间: {self._format_time(total_time)}")
        print(f"   平均CPU使用: {report['avg_cpu_usage']:.1f}%")
        print(f"   平均内存使用: {report['avg_memory_usage']:.1f}%")
        if 'avg_gpu_load' in report:
            print(f"   平均GPU负载: {report['avg_gpu_load']:.1f}%")
    
    def _format_time(self, seconds: float) -> str:
        """格式化时间显示"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def force_kill_process(self):
        """强制终止进程"""
        if self.process and self.process.poll() is None:
            try:
                print(f"\n🛑 强制终止进程 {self.process_name} (PID: {self.process.pid})")
                self.process.terminate()
                
                # 等待5秒，如果还没结束就强制杀死
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print("   进程未响应SIGTERM，发送SIGKILL")
                    self.process.kill()
                    self.process.wait()
                
                print("   进程已终止")
                return True
            except Exception as e:
                print(f"   终止进程失败: {e}")
                return False
        return False


class LogTailMonitor:
    """日志尾部实时监控器"""
    
    def __init__(self, log_file: str, keywords: List[str] = None):
        self.log_file = log_file
        self.keywords = keywords or ["epoch", "loss", "进度", "完成", "错误", "异常"]
        self.monitoring = False
        self.monitor_thread = None
        self.last_position = 0
    
    def start_monitoring(self):
        """开始监控日志"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._tail_log, daemon=True)
        self.monitor_thread.start()
        print(f"开始监控日志文件: {self.log_file}")
    
    def stop_monitoring(self):
        """停止监控日志"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def _tail_log(self):
        """实时跟踪日志文件"""
        while self.monitoring:
            try:
                if os.path.exists(self.log_file):
                    with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        f.seek(self.last_position)
                        new_lines = f.readlines()
                        self.last_position = f.tell()
                        
                        for line in new_lines:
                            line = line.strip()
                            if line and any(keyword in line.lower() for keyword in self.keywords):
                                timestamp = datetime.now().strftime("%H:%M:%S")
                                print(f"\n📝 [{timestamp}] {line}")
                
                time.sleep(2)  # 每2秒检查一次
                
            except Exception as e:
                print(f"\n日志监控错误: {e}")
                time.sleep(5)


def run_with_monitoring(command: List[str], log_file: str, process_name: str, 
                       timeout: int = 3600, enable_log_tail: bool = True) -> Dict:
    """
    运行命令并进行实时监控
    
    Args:
        command: 要执行的命令列表
        log_file: 日志文件路径
        process_name: 进程名称
        timeout: 超时时间（秒）
        enable_log_tail: 是否启用日志尾部监控
    
    Returns:
        执行结果字典
    """
    print(f"\n🚀 启动监控执行: {process_name}")
    print(f"   命令: {' '.join(command)}")
    print(f"   日志: {log_file}")
    print(f"   超时: {timeout}秒")
    
    # 创建监控器
    monitor = ProcessMonitor(process_name, log_file)
    log_monitor = None
    
    if enable_log_tail:
        log_monitor = LogTailMonitor(log_file)
        log_monitor.start_monitoring()
    
    result = {
        'success': False,
        'return_code': -1,
        'duration': 0,
        'timeout_occurred': False,
        'manually_killed': False,
    }
    
    start_time = time.time()
    
    try:
        # 启动进程
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # 开始监控
        monitor.start_monitoring(process)
        
        print(f"   进程已启动 (PID: {process.pid})")
        print(f"   按 Ctrl+C 可以强制终止进程")
        print("   " + "="*50)
        
        # 等待进程完成或超时
        try:
            process.wait(timeout=timeout)
            result['return_code'] = process.returncode
            result['success'] = process.returncode == 0
            
        except subprocess.TimeoutExpired:
            print(f"\n⏰ 进程超时 ({timeout}秒)，正在终止...")
            result['timeout_occurred'] = True
            monitor.force_kill_process()
            
        except KeyboardInterrupt:
            print(f"\n⌨️  用户中断，正在终止进程...")
            result['manually_killed'] = True
            monitor.force_kill_process()
    
    except Exception as e:
        print(f"\n❌ 执行过程中发生错误: {e}")
        result['error'] = str(e)
    
    finally:
        end_time = time.time()
        result['duration'] = end_time - start_time
        
        # 停止监控
        monitor.stop_monitoring()
        if log_monitor:
            log_monitor.stop_monitoring()
        
        print(f"\n🏁 进程执行完成")
        print(f"   耗时: {monitor._format_time(result['duration'])}")
        print(f"   成功: {result['success']}")
        print(f"   返回码: {result['return_code']}")
    
    return result


if __name__ == "__main__":
    # 测试监控功能
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python real_time_monitor.py <command> [args...]")
        sys.exit(1)
    
    command = sys.argv[1:]
    process_name = os.path.basename(command[0])
    log_file = f"{process_name}_test.log"
    
    result = run_with_monitoring(command, log_file, process_name, timeout=60)
    print(f"\n最终结果: {result}") 