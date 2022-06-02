import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../rpc_stubs'))


from runtime.rpc_stubs.worker_to_master_pb2 import RegisterWorkerRequest, RegisterWorkerResponse, DoneResponse
from runtime.rpc_stubs.worker_to_master_pb2_grpc import WorkerToMasterServicer
import runtime.rpc_stubs.worker_to_master_pb2_grpc as w2m_rpc

import grpc
from concurrent import futures


class MasterServerForWorker(WorkerToMasterServicer):
    def __init__(self, logger, callbacks) -> None:
        super().__init__()

        self._logger = logger
        self._callbacks = callbacks
    

    def RegisterWorker(self, request: RegisterWorkerRequest, context) -> RegisterWorkerResponse:
        # return super().RegisterWorker(request, context)
        assert 'RegisterWorker' in self._callbacks
        register_worker_impl = self._callbacks['RegisterWorker']

        success, worker_id = register_worker_impl(request.worker_ip, request.worker_port, request.num_gpus)
        response = RegisterWorkerResponse(success=success, worker_id=worker_id)
        
        return response
    

    def Done(self, request, context):
        # return super().Done(request, context)
        assert 'Done' in self._callbacks
        done_impl = self._callbacks['Done']

        success = done_impl(request.job_id, request.job_counter, request.worker_id, request.gpus, request.returncode)
        response = DoneResponse(success=success)

        return response


def serve(port, logger, callbacks):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    w2m_rpc.add_WorkerToMasterServicer_to_server(MasterServerForWorker(logger, callbacks), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()

    logger.info(f'controller, rpc, start, server @ {port}')
    
    server.wait_for_termination()