import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../rpc_stubs'))

from runtime.rpc_stubs.master_to_worker_pb2 import ExecuteRequest, KillRequest, ExitCommandRequest, GetUtilRequest
import runtime.rpc_stubs.master_to_worker_pb2_grpc as m2w_rpc

import grpc
from logging import Logger


class MasterClientForWorker(object):
    def __init__(self, logger : Logger, worker_id, worker_ip, worker_port) -> None:
        super().__init__()

        self._worker_id = worker_id
        self._worker_ip = worker_ip
        self._worker_port = worker_port
        channel = grpc.insecure_channel(self.addr)
        self._stub = m2w_rpc.MasterToWorkerStub(channel)
        self._logger = logger
    

    @property
    def addr(self):
        return f'{self._worker_ip}:{self._worker_port}'
    

    def execute(self, job_info):
        self._logger.info(f'controller, execute, {job_info.job_id} - {job_info.job_counter} @ {self._worker_id}-{job_info.node_id}, {job_info.gpus}')
        request = ExecuteRequest()
        request.job_info.num = int(job_info.num)
        request.job_info.node_id.extend(list(job_info.node_id))
        request.job_info.job_id.extend(list(job_info.job_id))
        request.job_info.job_name.extend(list(job_info.job_name))
        request.job_info.batch_size.extend(list(job_info.batch_size))
        request.job_info.iterations.extend(list(job_info.iterations))
        request.job_info.gpus = job_info.gpus
        request.job_info.job_counter.extend(list(job_info.job_counter))
        request.job_info.num_gpu = int(job_info.num_gpu)
        response = self._stub.Execute(request)
        assert response.success == True
    

    def kill(self, job_info):
        self._logger.info(f'controller, kill, {job_info.job_id} - {job_info.job_counter} @ {self._worker_id}-{job_info.node_id}, {job_info.gpus}')
        request = KillRequest()
        request.job_info.num = int(job_info.num)
        request.job_info.node_id.extend(list(job_info.node_id))
        request.job_info.job_id.extend(list(job_info.job_id))
        request.job_info.job_name.extend(list(job_info.job_name))
        request.job_info.batch_size.extend(list(job_info.batch_size))
        request.job_info.iterations.extend(list(job_info.iterations))
        request.job_info.gpus = job_info.gpus
        request.job_info.job_counter.extend(list(job_info.job_counter))
        request.job_info.num_gpu = int(job_info.num_gpu)
        response = self._stub.Kill(request)
        assert response.success == True

    def exit_command(self):
        self._logger.info(f'controller ask worker {self._worker_id} to exit')
        request = ExitCommandRequest()
        response = self._stub.ExitCommand(request)
        assert response.success == True
    
    def get_util(self, secs):
        request = GetUtilRequest(secs=secs)
        response = self._stub.GetUtil(request)
        return response.gpu_util, response.cpu_util, response.io_read

