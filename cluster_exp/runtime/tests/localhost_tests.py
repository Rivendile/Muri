import unittest
import socket
import sys
import os
import time
import threading
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import worker
import master

LOCALHOST = "127.0.0.1"
MASTER_PORT = 6888
WORKER_PORT1 = 9000
WORKER_PORT2 = 9001
WORKER_PORT3 = 9002
TRACE_FILE = "../../traces/fake.txt"

class Worker2MasterTests(unittest.TestCase):
    # worker_client -> master_server
    def test_register_and_done(self):
        # start master server in the background at localhost:6888
        m = master.Master(MASTER_PORT)
        time.sleep(2)

        # initialize two worker client
        client0 = worker.Worker(LOCALHOST, MASTER_PORT, LOCALHOST, WORKER_PORT1, 2)
        client1 = worker.Worker(LOCALHOST, MASTER_PORT, LOCALHOST, WORKER_PORT2, 2)

        # register a worker with 2 GPUs
        ret = client0.register()
        # register successfully with worker_id 0
        self.assertEqual(0, ret)

        #register a worker with 5 GPUs
        ret = client1.register()
        # worker_id 1
        self.assertEqual(1, ret)

        #clinet 0 job 0 request for fast forward
        client0._worker_rpc_client.report_stable(0)
        m.fast_forward(1234)
        client1._worker_rpc_client.report_stable(0)

        # client0 has done job 7
        client0._worker_rpc_client.done(7)
        # client1 has done job 12
        client1._worker_rpc_client.done(12)
        # client0 has done job 999
        client0._worker_rpc_client.done(999)
        # client1 has done job 777
        client1._worker_rpc_client.done(777)

class Master2WorkerTests(unittest.TestCase):
    def setUp(self):
        self.client0 = worker.Worker(LOCALHOST, MASTER_PORT, LOCALHOST, WORKER_PORT1, 2)
        self.client1 = worker.Worker(LOCALHOST, MASTER_PORT, LOCALHOST, WORKER_PORT2, 2)
        self.master = master.Master(MASTER_PORT)
        self.client0.register()
        self.client1.register()

    # master_client => worker_server
    def test_run_and_kill_and_update(self):
        job_description1 = ["fake", 0, LOCALHOST, 3] # start fake.py with job_id = 0 on localhost
        job_description2 = ["fake", 1, LOCALHOST, 1] # start fake.py with job_id = 1 on localhost
        job_description3 = ["fake", 2, LOCALHOST, 1] # start fake.py with job_id = 2 on localhost

        self.master._workers[0].caller.run_job(job_description1)
        self.master._workers[1].caller.run_job(job_description2)
        self.master._workers[1].caller.run_job(job_description3)
        time.sleep(5)
        job_description1[-1] = 8 
        self.master._workers[0].caller.update_job(job_description1)
        time.sleep(5)
        self.master._workers[0].caller.kill_job(0) # kill job 0
        time.sleep(10)
        self.master._workers[1].caller.kill_job(1) # kill job 1
    
    def tearDown(self):
        # this will kill job 2
        self.master.shutdown()

if __name__ == "__main__":
    unittest.main()