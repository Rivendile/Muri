import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../rpc_stubs'))

from runtime.rpc_stubs.scheduler_to_trainer_pb2 import QueryStatsRequest
import runtime.rpc_stubs.scheduler_to_trainer_pb2_grpc as s2t_rpc

import grpc
from logging import Logger


class SchedulerClientForTrainer(object):
    def __init__(self, logger : Logger, job_id, trainer_ip, trainer_port) -> None:
        super().__init__()

        self._job_id = job_id
        self._trainer_ip = trainer_ip
        self._trainer_port = trainer_port
        channel = grpc.insecure_channel(self.addr)
        self._stub = s2t_rpc.SchedulerToTrainerStub(channel)
        self._logger = logger
    

    @property
    def addr(self):
        return f'{self._trainer_ip}:{self._trainer_port}'
    

    def query_stats(self):
        self._logger.info(f'scheduler, query, job {self._job_id}')
        request = QueryStatsRequest()
        response = self._stub.QueryStats(request)
        
        return response.finished_iterations