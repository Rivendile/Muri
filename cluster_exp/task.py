from runtime.rpc_stubs.master_to_worker_pb2 import JobInfo

import subprocess
import os
import utils


class Task(object):
    def __init__(self, job_info: JobInfo, scheduler_ip, trace_name, this_dir) -> None:
        super().__init__()

        self._job_num = job_info.num
        self._node_id = list(job_info.node_id)
        self._job_id = job_info.job_id
        self._job_name = job_info.job_name
        self._batch_size = job_info.batch_size
        self._iterations = job_info.iterations
        self._gpus = job_info.gpus
        self._scheduler_ip = scheduler_ip
        self._num_gpu = job_info.num_gpu
        self._this_dir = this_dir
        self._job_counter = job_info.job_counter
        self._trace_name = trace_name
    

    def get_idle_port(self):
        return 9013 + 8*min(self._node_id) + int(self._gpus.split(',')[0])
        # return utils.find_free_port()


    @staticmethod
    def test_kill_restart():
        bash_cmd = 'nvidia-smi; sleep 2m; date'
        return bash_cmd


    def real_job(self):
        bash_cmd = f'bash {self._this_dir}/workloads/run.sh'
        for i in range(self._job_num):
            bash_cmd += f' {self._job_name[i]} {self._batch_size[i]} 0 2 -1 {self._iterations[i]} {self._job_id[i]} {self._job_counter[i]}'
        bash_cmd += f' {self._num_gpu}'
        bash_cmd += f' --scheduler-ip {self._scheduler_ip}'
        bash_cmd += f' --trainer-port {self.get_idle_port()} --this-dir {self._this_dir}/workloads'
        return bash_cmd

    def run(self):
        bash_cmd = ''
        # if self._job_name == 'test_kill_restart':
        #     bash_cmd = self.test_kill_restart()
        # else:
        bash_cmd = self.real_job()

        cmd = bash_cmd.split()

        hostfile_dir = self._this_dir+'/workloads/hostfiles'
        assert os.path.exists(hostfile_dir)
        hostfile_list = [f'worker-{node_id}\n' for node_id in self._node_id]
        ch = '-'
        job_id_str = ch.join([str(x) for x in list(self._job_id)])
        job_counter_str = ch.join([str(x) for x in list(self._job_counter)])
        # print(self._iterations)
        with open(hostfile_dir+f'/hostfile-[{job_id_str}]-[{job_counter_str}]', 'w') as f:
            f.writelines(hostfile_list)

        environ_dict = dict(os.environ)
        environ_dict['CUDA_VISIBLE_DEVICES'] = self._gpus
        with open(self.log_path, 'w+') as f:
            self._handler = subprocess.Popen(
                cmd, 
                stdout=f,
                stderr=f,
                env=environ_dict,
            )

        return cmd
    

    def terminate(self):
        self._handler.terminate()
    
    def wait(self):
        self._handler.wait()
    

    @property
    def return_code(self):
        return self._handler.poll()

    @property
    def pid(self):
        return self._handler.pid


    @property
    def log_path(self):
        if not os.path.exists(f'{self._trace_name}/'):
            os.makedirs(f'{self._trace_name}/')
        path = ''
        for i in range(self._job_num):
            if i==0:
                path = f'{self._trace_name}/{self._job_id[i]}-{self._job_counter[i]}-{self._job_name[i]}'
            else:
                path += f'_{self._job_id[i]}-{self._job_counter[i]}-{self._job_name[i]}'
        return path + '.txt'
