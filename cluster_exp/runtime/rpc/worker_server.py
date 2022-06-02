import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../rpc_stubs'))

from runtime.rpc_stubs.master_to_worker_pb2 import ExecuteResponse, KillResponse, ExitCommandResponse, GetUtilResponse
from runtime.rpc_stubs.master_to_worker_pb2_grpc import MasterToWorkerServicer
import runtime.rpc_stubs.master_to_worker_pb2_grpc as m2w_rpc

import grpc
from concurrent import futures


class WorkerServerForMaster(MasterToWorkerServicer):
    def __init__(self, logger, callbacks) -> None:
        super().__init__()
        self._logger = logger
        self._callbacks = callbacks
    

    def Execute(self, request, context):
        # return super().Execute(request, context)
        assert 'Execute' in self._callbacks
        execute_impl = self._callbacks['Execute']

        success = execute_impl(request.job_info)
        response = ExecuteResponse(success=success)

        return response
    

    def Kill(self, request, context):
        # return super().Kill(request, context)
        assert 'Kill' in self._callbacks
        kill_impl = self._callbacks['Kill']

        success = kill_impl(request.job_info)
        response = KillResponse(success=success)

        return response

    def ExitCommand(self, request, context):
        assert 'ExitCommand' in self._callbacks
        exit_command_impl = self._callbacks['ExitCommand']

        success = exit_command_impl()
        response = ExitCommandResponse(success=success)

        return response

    def GetUtil(self, request, context):
        assert 'GetUtil' in self._callbacks
        get_util_impl = self._callbacks['GetUtil']

        gpu_util, cpu_util, io_read = get_util_impl(request.secs)
        response = GetUtilResponse(gpu_util=gpu_util, cpu_util=cpu_util, io_read=io_read)

        return response


def serve(port, logger, callbacks):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    m2w_rpc.add_MasterToWorkerServicer_to_server(WorkerServerForMaster(logger, callbacks), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()

    logger.info(f'worker, rpc, start, server @ {port}')
    
    server.wait_for_termination()