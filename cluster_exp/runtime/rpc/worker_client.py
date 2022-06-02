import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../rpc_stubs'))

from runtime.rpc_stubs.worker_to_master_pb2 import RegisterWorkerRequest, DoneRequest
import runtime.rpc_stubs.worker_to_master_pb2_grpc as w2m_rpc

import grpc


class WorkerClientForMaster(object):
    def __init__(self, logger, master_ip, master_port) -> None:
        super().__init__()

        self._logger = logger

        self._master_ip = master_ip
        self._master_port = master_port

        channel = grpc.insecure_channel(self.addr)
        self._stub = w2m_rpc.WorkerToMasterStub(channel)
    

    @property
    def addr(self):
        return f'{self._master_ip}:{self._master_port}'

    
    def register_worker(self, worker_ip, worker_port, num_gpus):
        request = RegisterWorkerRequest(worker_ip=worker_ip, worker_port=worker_port, num_gpus=num_gpus)

        try:
            response = self._stub.RegisterWorker(request)
            self._logger.info(f'{response.worker_id}, register, {response.success}')
            return response.success, response.worker_id
        except Exception as e:
            self._logger.info(f'worker, register, fail, {e}')
            return False, None
    
    def done(self, job_id, job_counter, worker_id, gpus, returncode):
        request = DoneRequest(job_id=job_id, job_counter=job_counter, worker_id=worker_id, gpus=gpus, returncode = returncode)
        response = self._stub.Done(request)

        self._logger.info(f'{worker_id}, done, {job_id} - {job_counter}, {gpus}, return code: {returncode}')
