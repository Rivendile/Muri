import unittest
import socket
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../rpc'))
from worker_server import WorkerRpcServer
from worker_client import WorkerRpcClient

class TestWorkerRpcServer(unittest.TestCase):
    def test_fetch_gpu_list(self):
        workerSever = WorkerRpcServer(None)
        tests = [0b1, 0, 0b101, 0b1111]
        tests_sol = [[0], [], [0, 2], [0, 1, 2, 3]]
        for test, sol in zip(tests, tests_sol):
            self.assertEqual(workerSever._fetch_GPU_list(test), sol)
    

if __name__ == "__main__":
    unittest.main()