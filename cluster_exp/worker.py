import argparse
from logging import root
import utils
import threading
import time
import subprocess
import os
import signal
import math

from runtime.rpc import worker_server, worker_client
from task import Task


class Worker(object):
    def __init__(self, master_ip, master_port, worker_ip, worker_port, gpus: str, trace_name, this_dir) -> None:
        super().__init__()

        self._logger = utils.make_logger(__name__)
        
        self._master_ip = master_ip
        self._master_port = master_port
        self._work_ip = worker_ip
        self._worker_port = worker_port
        self._worker_id = None
        self._trace_name = trace_name
        self._this_dir = this_dir
        self._check_task_flag = True
        
        self._gpus = gpus.split(',')
        self._num_gpus = len(self._gpus)

        self._client_for_master = worker_client.WorkerClientForMaster(self._logger, self._master_ip, self._master_port)

        self._server_for_master = self.make_server_for_master(self._worker_port)
        
        self._tasks = dict()

        self.register()
    

    def register(self):
        while True:
            success, worker_id = self._client_for_master.register_worker(self._work_ip, self._worker_port, self._num_gpus)

            if success == True:
                self._worker_id = worker_id
                break
            
            time.sleep(5)
    

    def check_tasks(self):
        while self._check_task_flag:
            finished_tasks = []
            error_tasks = []

            for (job_id, job_counter), (task, job_info) in self._tasks.items():
                if task.return_code == None: # the pid of task is the pid of mpirun
                    continue
                
                self._client_for_master.done(job_id, job_counter, self._worker_id, task._gpus, task.return_code)
                finished_tasks.append((job_id, job_counter))
                if task.return_code != 0:
                    with open(f'{self._this_dir}/workloads/test_{job_id}.txt', 'r') as f:
                        error_text = f.read()
                        self._logger.info(f'error info: {job_id} {job_counter} \n'+error_text)
            
            for (job_id, job_counter) in finished_tasks:
                self._tasks.pop((job_id, job_counter))
            
            time.sleep(2)
    

    def make_server_for_master(self, port: int):
        callbacks = {
            'Execute' : self._execute_impl,
            'Kill' : self._kill_impl,
            'ExitCommand' : self._exit_command_impl,
            'GetUtil' : self._get_util_impl,
        }
        # GetUtil is deprecated

        server_thread = threading.Thread(
            target=worker_server.serve,
            args=(port, self._logger, callbacks))
        server_thread.setDaemon(True)
        server_thread.start()

        return server_thread
    

    def _execute_impl(self, job_info) -> bool:
        success = True

        task = Task(job_info, self._master_ip, self._trace_name, self._this_dir)
        cmd = task.run()
        self._tasks[(max(task._job_id), max(task._job_counter))] = (task, job_info)

        self._logger.info(f'{self._worker_id}, execute, {task._job_id} - {task._job_counter}, {task._gpus}, {" ".join(cmd)}')

        # print(success)

        return success
    

    def _kill_impl(self, job_info) -> bool:
        job_id = max(job_info.job_id)
        job_counter = max(job_info.job_counter)

        if (job_id, job_counter) not in self._tasks:
            return False

        task, _ = self._tasks.pop((job_id, job_counter))
        task.terminate()
        task.wait()
        self._logger.info(f'{self._worker_id}, kill, {job_id} - {job_counter}, {job_info.gpus}')
        
        with open(f'{self._this_dir}/workloads/test_{job_id}.txt', 'r') as f:
            kill_text = f.read()
            self._logger.info(f'kill info: {job_id} {job_counter} \n'+kill_text)

        return True

    def _exit_command_impl(self):
        self._logger.info(f'{self._worker_id} exit')
        self._check_task_flag = False
        return True

    # Deprecated
    def _get_util_impl(self, secs):
        # prepare
        device_list = range(self._num_gpus)
        process_list = []
        os.system("rm -rf profiling*.xml")
        os.system("rm -rf profiling*.out")

        # start subprocess
        # gpu
        for device in device_list:
            filename = "profiling" + str(device) + ".xml"
            command = "exec nvidia-smi -q -i " + str(device) + " -x -l 1 -f " + filename
            process_list.append(subprocess.Popen(command, shell=True))
        # cpu
        cpu_command = "exec top -d 1 -bn " + str(secs) + " | grep Cpu > profiling_cpu.out"
        cpu_process = subprocess.Popen(cpu_command, shell=True)
        # io
        io_command = "exec iostat -d 1 " + str(secs) + " | grep nvme > profiling_disk.out"
        io_process = subprocess.Popen(io_command, shell=True)

        # wait
        time.sleep(secs)
        for process in process_list:
            process.send_signal(signal.SIGINT)
            process.terminate()
            process.wait()
        cpu_process.wait()
        io_process.wait()

        # handle results
        useful_ratio = 0
        gpu_utils = []
        for device in device_list:
            filename = "profiling" + str(device) + ".xml"
            memory_usage, utilization = utils.parse_xml(filename)
            for i in range(len(memory_usage)):
                memory_usage[i] = int(memory_usage[i].split(' ')[0])
                utilization[i] = int(utilization[i].split(' ')[0])
            self._logger.info(f'{memory_usage}, {utilization}')
            sorted_memory_usages = sorted(memory_usage)
            gpu_util_device = 0
            gpu_util_cnt = 0
            for i in range(len(memory_usage)):
                if math.isclose(memory_usage[i], sorted_memory_usages[-2], rel_tol=1e-1):
                    gpu_util_device += utilization[i]
                    gpu_util_cnt += 1
            gpu_utils.append(gpu_util_device/gpu_util_cnt)
            self._logger.info(f'gpu util of device {device}: {memory_usage} {utilization} {gpu_utils[-1]}')
        gpu_util = sum(gpu_utils)/len(gpu_utils)
        self._logger.info(f'gpu util: {gpu_utils} {gpu_util}')

        
        cpu_util_list = []
        util_str_list = open("profiling_cpu.out", "r").read().split('\n')
        for i in range(secs):
            idle = float(util_str_list[i].split()[7])
            cpu_util_list.append(round(100.0 -idle, 3))
        cpu_util = sum(cpu_util_list)/len(cpu_util_list)
        self._logger.info(f'cpu util: {cpu_util_list}, {cpu_util}')
        
        io_util_list = []
        util_str_list = open("profiling_disk.out", "r").read().split('\n')
        for i in range(secs):
            kB_read = float(util_str_list[i].split()[2])
            io_util_list.append(kB_read)
        io_read = sum(io_util_list[1:])/(len(io_util_list)-1)
        self._logger.info(f'io read: {io_util_list} {io_read}')


        return gpu_util, cpu_util, io_read

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--master_ip', type=str, required=True)
    parser.add_argument('--master_port', type=int, default=9012)
    parser.add_argument('--worker_port', type=int, default=9001)
    parser.add_argument('--gpus', type=str, default='0,1,2,3,4,5,6,7')
    parser.add_argument('--trace_name', type=str, default='test')
    parser.add_argument('--this-dir', type=str, default='./')

    args = parser.parse_args()

    worker_ip = utils.get_host_ip()
    worker = Worker(args.master_ip, args.master_port, worker_ip, args.worker_port, args.gpus, args.trace_name, args.this_dir)
    worker.check_tasks()