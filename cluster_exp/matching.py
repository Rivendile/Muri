from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import numpy
import math
import flags
import copy
import itertools
FLAGS = flags.FLAGS

def my_cmp(x,y):
    if math.isclose(x, y, rel_tol=1e-5):
        return 0
    if x>y:
        return 1
    else:
        return 0

def swap(x,y):
    return y,x

class _Packing(object):
    class _MiniJob(object):
        '''
        mini job for class:packing
        '''
        def __init__(self, rjob):
            self.num_gpu = rjob['num_gpu']
            self.resource_time = rjob['resource_time']
            self.job_idx = rjob['job_idx']
            self.iteration_time = rjob['iteration_time']
            self.sort_val = rjob['sort_val']
        def calc_iter_time(self):
            return sum(self.resource_time)

    def __init__(self, rjob):
        self.packing_jobs = list()
        job_tmp = self._MiniJob(rjob)
        self.packing_jobs.append(job_tmp)
        self.num_gpu = job_tmp.num_gpu
        self.best_permutation = None

    def add_job(self, rjob):
        self.packing_jobs.extend(rjob.packing_jobs)

    def calc_iteration_time(self, packing=None, ordering=1):
        TT_all = float("inf")
        if packing!=None:
            jobs = self.packing_jobs + packing.packing_jobs
        else:
            jobs = self.packing_jobs
        if len(jobs) == 1:
            self.best_permutation = (jobs[0],)
            return jobs[0].iteration_time
        if ordering==1: # best ordering
            jobs_permutation = itertools.permutations(jobs)
            for permutation in jobs_permutation:
                TT = 0.0
                if FLAGS.multi_resource==4 and len(jobs[0].resource_time)==3:
                    round = 2
                    fi = [0 for _ in range(4*round+1)]
                    num_job = len(jobs)
                    for i in range(4*round):
                        if (i+1)%4<num_job:
                            fi[i+1] = max(fi[i+1], fi[i]+permutation[(i+1)%4].resource_time[2])
                        if (i+2)%4<num_job:
                            fi[i+1] = max(fi[i+1], fi[i]+permutation[(i+2)%4].resource_time[1])
                        if i>0 and (i+3)%4<num_job:
                            fi[i+1] = max(fi[i+1], fi[i-1]+permutation[(i+3)%4].resource_time[0])
                        if i>=4 and math.isclose(fi[i+1]-fi[i-3], fi[i]-fi[i-4], rel_tol = 1e-3):
                            TT = fi[i+1]-fi[i-3]
                            break
                    if not(TT>0):
                        TT = fi[4*round]-fi[4*round-4]
                    assert TT>0
                else:
                    for i in range(FLAGS.multi_resource):
                        max_num = 0.0
                        for idx, val in enumerate(permutation):
                            # print(i, idx, FLAGS.multi_resource, (i-idx+FLAGS.multi_resource)%FLAGS.multi_resource)
                            max_num = max(max_num, val.resource_time[(i-idx+FLAGS.multi_resource)%FLAGS.multi_resource])
                        TT += max_num
                if TT < TT_all:
                    TT_all = TT
                    self.best_permutation = copy.deepcopy(permutation)
            if len(jobs)==2 and FLAGS.multi_resource==4 and len(jobs[0].resource_time)==3:
                TT = max(jobs[0].resource_time[0], jobs[1].resource_time[1]+jobs[1].resource_time[2])+max(jobs[1].resource_time[0], jobs[0].resource_time[1]+jobs[0].resource_time[2])
                if TT < TT_all:
                    TT_all = TT
                    self.best_permutation = (jobs[0], None, jobs[1])
        elif ordering==2: # worst ordering
            TT_all = 0.0
            jobs_permutation = itertools.permutations(jobs)
            for permutation in jobs_permutation:
                TT = 0.0
                if FLAGS.multi_resource==4 and len(jobs[0].resource_time)==3:
                    round = 2
                    fi = [0 for _ in range(4*round+1)]
                    num_job = len(jobs)
                    for i in range(4*round):
                        if (i+1)%4<num_job:
                            fi[i+1] = max(fi[i+1], fi[i]+permutation[(i+1)%4].resource_time[2])
                        if (i+2)%4<num_job:
                            fi[i+1] = max(fi[i+1], fi[i]+permutation[(i+2)%4].resource_time[1])
                        if i>0 and (i+3)%4<num_job:
                            fi[i+1] = max(fi[i+1], fi[i-1]+permutation[(i+3)%4].resource_time[0])
                        if i>=4 and math.isclose(fi[i+1]-fi[i-3], fi[i]-fi[i-4], rel_tol = 1e-3):
                            TT = fi[i+1]-fi[i-3]
                            break
                    if not(TT>0):
                        TT = fi[4*round]-fi[4*round-4]
                    assert TT>0
                else:
                    for i in range(FLAGS.multi_resource):
                        max_num = 0.0
                        for idx, val in enumerate(permutation):
                            # print(i, idx, FLAGS.multi_resource, (i-idx+FLAGS.multi_resource)%FLAGS.multi_resource)
                            max_num = max(max_num, val.resource_time[(i-idx+FLAGS.multi_resource)%FLAGS.multi_resource])
                        TT += max_num
                if TT > TT_all:
                    TT_all = TT
                    self.best_permutation = copy.deepcopy(permutation)
            if len(jobs)==2 and FLAGS.multi_resource==4 and len(jobs[0].resource_time)==3:
                TT = max(jobs[0].resource_time[0], jobs[1].resource_time[1]+jobs[1].resource_time[2])+max(jobs[1].resource_time[0], jobs[0].resource_time[1]+jobs[0].resource_time[2])
                if TT > TT_all:
                    TT_all = TT
                    self.best_permutation = (jobs[0], None, jobs[1])
        else: # no ordering
            permutation = sorted(jobs, key=lambda x: x.job_idx)
            TT = 0.0
            if FLAGS.multi_resource==4 and len(jobs[0].resource_time)==3:
                round = 2
                fi = [0 for _ in range(4*round+1)]
                num_job = len(jobs)
                for i in range(4*round):
                    if (i+1)%4<num_job:
                        fi[i+1] = max(fi[i+1], fi[i]+permutation[(i+1)%4].resource_time[2])
                    if (i+2)%4<num_job:
                        fi[i+1] = max(fi[i+1], fi[i]+permutation[(i+2)%4].resource_time[1])
                    if i>0 and (i+3)%4<num_job:
                        fi[i+1] = max(fi[i+1], fi[i-1]+permutation[(i+3)%4].resource_time[0])
                    if i>=4 and math.isclose(fi[i+1]-fi[i-3], fi[i]-fi[i-4], rel_tol = 1e-3):
                        TT = fi[i+1]-fi[i-3]
                        break
                if not(TT>0):
                    TT = fi[4*round]-fi[4*round-4]
                assert TT>0
            else:
                for i in range(FLAGS.multi_resource):
                    max_num = 0.0
                    for idx, val in enumerate(permutation):
                        # print(i, idx, FLAGS.multi_resource, (i-idx+FLAGS.multi_resource)%FLAGS.multi_resource)
                        max_num = max(max_num, val.resource_time[(i-idx+FLAGS.multi_resource)%FLAGS.multi_resource])
                    TT += max_num
            if TT < TT_all:
                TT_all = TT
                self.best_permutation = copy.deepcopy(permutation)
        return TT_all

    def calc_used_ratio(self, packing, ordering=1):
        jobs = self.packing_jobs + packing.packing_jobs
        TT = self.calc_iteration_time(packing, ordering)
        used_time = [sum(i.resource_time) for i in jobs]
        used_time_sum = sum(used_time)

        return used_time_sum/(TT*FLAGS.multi_resource)
    
    def calc_weight(self, packing):
        jobs = self.packing_jobs + packing.packing_jobs
        TT = self.calc_iteration_time(packing)
        min_val = float('inf')
        minx = 0
        for rjob in jobs:
            if rjob.sort_val<min_val:
                min_val = rjob.sort_val
                minx = rjob.iteration_time
        if TT > len(jobs)*minx:
            return 0
        used_time = [sum(i.resource_time) for i in jobs]
        used_time_sum = sum(used_time)

        return used_time_sum/(TT*FLAGS.multi_resource)

    
    def calc_self_used_ratio(self):
        jobs = self.packing_jobs
        TT = self.calc_iteration_time()
        
        used_time = [sum(i.resource_time) for i in jobs]
        used_time_sum = sum(used_time)

        return used_time_sum/(TT*FLAGS.multi_resource)

class _Blossom_Same(object):
    '''
    matching algorithm for multi-resource packing
    Blossom algorithm for general graph matching
    only match jobs with the same #GPU
    '''
    def __init__(self):
        self.run_jobs_list = list()

        # params for matching
        # note that the node id for blossom algo is 1..n
        # the node id for scheduler is 0..n-1
        #int
        self.match = list()
        self.st = list()
        self.node_cnt = 0
        self.flower = list()
        self.flower_from = list()
        self.S = list()
        self.slack = list()
        self.n_x = 0
        self.q = list()
        self.pa = list()
        self.t = 0
        self.vis=list()
        self.required_gpu = 0
        #double
        self.lab = list()
        #dict
        self.edge = list()
    
    # slack value of e
    def dist(self, e):
        return self.lab[e["u"]]+self.lab[e["v"]]-self.edge[e["u"]][e["v"]]["w"]*2.0

    # set the node u to be slack of x if u has smaller dist to x
    def update_slack(self, u, x):
        if self.slack[x]==0 or self.dist(self.edge[u][x])<self.dist(self.edge[self.slack[x]][x]):
            self.slack[x] = u 
    
    # set the node u which has smallest dist to x
    def set_slack(self, x):
        self.slack[x] = 0
        for u in range(1, self.node_cnt+1):
            if self.edge[u][x]["w"]>0 and self.st[u]!=x and self.S[self.st[u]]==0:
                self.update_slack(u, x)

    # push x and its flower to the queue
    def q_push(self, x):
        if x<=self.node_cnt:
            self.q.append(x)
        else:
            for i in self.flower[x]:
                self.q_push(i)

    # set st of x and nodes in flower x to be b
    def set_st(self, x, b):
        self.st[x] = b 
        if x<self.node_cnt:
            return
        for i in self.flower[x]:
            self.set_st(i, b)
    
    # get the relative position of b in the flower and 
    # reverse the sequence of flower b if position is odd.
    def get_pr(self, b, xr):
        pr = None
        for i, value in enumerate(self.flower[b]):
            if value==xr:
                pr = i
                break
        assert pr!=None
        if pr%2==1:
            tmp_flower = [self.flower[b][0]]+self.flower[b][1:][::-1]
            self.flower[b] = tmp_flower
            return len(self.flower[b])-pr
        else:
            return pr
    
    # set the match of u to be v; specially deal with u when u is a flower
    def set_match(self, u, v):
        self.match[u] = self.edge[u][v]["v"]
        if u<= self.node_cnt:
            return
        e = self.edge[u][v]
        xr = self.flower_from[u][e["u"]]
        pr = self.get_pr(u, xr)
        for i in range(pr):
            self.set_match(self.flower[u][i], self.flower[u][i^1])
        self.set_match(xr, v)
        tmp_flower = self.flower[u][pr:]+self.flower[u][:pr]
        self.flower[u] = tmp_flower
    
    # augment -- swap the match point on the path
    def augment(self, u, v):
        xnv = self.st[self.match[u]]
        self.set_match(u,v)
        if xnv==0:
            return
        self.set_match(xnv, self.st[self.pa[xnv]])
        self.augment(self.st[self.pa[xnv]], xnv)

    # get the lca of u and v
    def get_lca(self, u, v):
        self.t += 1
        while u!=0 or v!=0:
            if u!=0:
                if self.vis[u]==self.t:
                    return u 
                self.vis[u] = self.t 
                u = self.st[self.match[u]]
                if u!=0:
                    u = self.st[self.pa[u]]
            u,v = swap(u,v)
        return 0
    
    # add a blossom
    def add_blossom(self, u, lca, v):
        b = self.node_cnt +1
        while b<=self.n_x and self.st[b]!=0:
            b+=1
        if b>self.n_x:
            self.n_x += 1
            # extend list for b
            assert len(self.flower)==b
            self.flower.append([lca])
            assert len(self.match)==b
            self.match.append(self.match[lca])
            for rlist in self.edge:
                assert len(rlist)==b 
                rlist.append({"u":0, "v":0, "w":0})
            assert len(self.edge)==b 
            self.edge.append([{"u":0, "v":0, "w":0} for _ in range(b+1)])
            assert len(self.st)==b 
            self.st.append(b)
            for rlist in self.flower_from:
                assert len(rlist)==b 
                rlist.append(0)
            assert len(self.flower_from)==b 
            self.flower_from.append([0 for _ in range(b+1)])
            assert len(self.S)==b 
            self.S.append(0)
            assert len(self.slack)==b 
            self.slack.append(0)
            assert len(self.pa)==b 
            self.pa.append(0)
            assert len(self.vis)==b 
            self.vis.append(0)
            assert len(self.lab)==b 
            self.lab.append(0)
        else:
            self.lab[b], self.S[b] = 0, 0
            self.match[b] = self.match[lca]
            self.flower[b] = [lca]
        
        x=u 
        while x!=lca:
            self.flower[b].append(x)
            y = self.st[self.match[x]]
            self.flower[b].append(y)
            self.q_push(y)
            x=self.st[self.pa[y]]
        
        # reverse 0..n-1 -> 0 n-1 n-2 .. 1
        tmp_flower = [self.flower[b][0]]+self.flower[b][1:][::-1]
        self.flower[b] = tmp_flower

        x=v
        while x!=lca:
            self.flower[b].append(x)
            y = self.st[self.match[x]]
            self.flower[b].append(y)
            self.q_push(y)
            x=self.st[self.pa[y]]
        
        self.set_st(b, b)
        for x in range(1, self.n_x+1):
            self.edge[b][x]["w"] = 0
            self.edge[x][b]["w"] = 0
        for x in range(1, self.node_cnt+1):
            self.flower_from[b][x] = 0
        for xs in self.flower[b]:
            for x in range(1, self.n_x+1):
                if my_cmp(self.edge[b][x]["w"], 0)==0 or self.dist(self.edge[xs][x])<self.dist(self.edge[b][x]):
                    self.edge[b][x] = copy.deepcopy(self.edge[xs][x])
                    self.edge[x][b] = copy.deepcopy(self.edge[x][xs])
            for x in range(1, self.node_cnt+1):
                if self.flower_from[xs][x]!=0:
                    self.flower_from[b][x]=xs
        
        self.set_slack(b)

    # expand operation
    # If label[b]=0, S[b]=1, then flower b can be expanded.
    def expand_blossom(self, b):
        for i in self.flower[b]:
            self.set_st(i, i)
        xr = self.flower_from[b][self.edge[b][self.pa[b]]["u"]]
        pr = self.get_pr(b, xr)
        for i in range(0, pr, 2):
            xs = self.flower[b][i]
            xns = self.flower[b][i+1]
            self.pa[xs]=self.edge[xns][xs]["u"]
            self.S[xs], self.S[xns] = 1, 0
            self.slack[xs] = 0
            self.set_slack(xns)
            self.q_push(xns)
        self.S[xr], self.pa[xr] = 1, self.pa[b]
        for i in range(pr+1, len(self.flower[b])):
            xs = self.flower[b][i]
            self.S[xs] = -1
            self.set_slack(xs)
        self.st[b] = 0

    # judge if a new augment path of blossom is found
    def on_found_edge(self, e):
        u = self.st[e["u"]]
        v = self.st[e["v"]]
        if self.S[v]==-1:
            self.pa[v]=e["u"]
            self.S[v]=1
            nu = self.st[self.match[v]]
            self.slack[v], self.slack[nu], self.S[nu] = 0, 0, 0
            self.q_push(nu)
        elif self.S[v]==0:
            lca = self.get_lca(u,v)
            if lca==0:
                self.augment(u,v)
                self.augment(v,u)
                return True
            else:
                self.add_blossom(u, lca, v)
        return False

    def print_log(self):
        print(self.S[1:], "S")
        print(self.pa[1:], "pa")
        print(self.slack[1:], "slack")
        print(self.st[1:], "st")
        print(self.lab[1:], "lab")
        print(self.match[1:], "match")
        print(self.vis[1:], "vis")
        for i in range(1, self.n_x+1):
            print("flower", i, ":", self.flower[i])

    # return true if an augment path is found
    def matching(self):
        self.S = [-1 for _ in range(self.n_x+1)]
        self.slack = [0 for _ in range(self.n_x+1)]
        self.q = list()
        for x in range(1, self.n_x+1):
            if self.st[x]==x and self.match[x]<=0:
                self.pa[x] = 0
                self.S[x] = 0
                self.q_push(x)
        if len(self.q) == 0:
            return False
        # print('bfs queue: ', self.q)
        while True:
            while len(self.q)>0:
                # print("before queue: ")
                # self.print_log()
                u = self.q[0]
                del self.q[0]
                if self.S[self.st[u]]==1:
                    continue
                for v in range(1, self.node_cnt+1):
                    if self.edge[u][v]["w"]>0 and self.st[u] != self.st[v]:
                        # print(u, v, self.dist(self.edge[u][v]))
                        if my_cmp(self.dist(self.edge[u][v]), 0.0)==0:
                            if self.on_found_edge(self.edge[u][v]):
                                # print("on_found_edge: ", u, v)
                                return True
                        else:
                            self.update_slack(u, self.st[v])

            # print("finish queue:")
            # self.print_log()
            # calc d
            d = float("inf")
            for b in range(self.node_cnt+1, self.n_x+1):
                if self.st[b]==b and self.S[b]==1:
                    d = min(d, self.lab[b]/2.0)
            for x in range(1, self.n_x+1):
                if self.st[x]==x and self.slack[x]!=0:
                    if self.S[x]==-1:
                        d = min(d, self.dist(self.edge[self.slack[x]][x]))
                    elif self.S[x]==0:
                        d = min(d, self.dist(self.edge[self.slack[x]][x])/2.0)
            # update label
            for u in range(1, self.node_cnt+1):
                if self.S[self.st[u]]==0:
                    if my_cmp(self.lab[u], d)<=0:
                        return False
                    self.lab[u] -= d
                elif self.S[self.st[u]]==1:
                    self.lab[u] += d
            for b in range(self.node_cnt+1, self.n_x+1):
                if self.st[b]==b:
                    if self.S[self.st[b]]==0:
                        self.lab[b] += d*2
                    elif self.S[self.st[b]]==1:
                        self.lab[b] -= d*2
            self.q = list()
            for x in range(1, self.n_x+1):
                if self.st[x]==x and self.slack[x]!=0 and self.st[self.slack[x]]!=x and my_cmp(self.dist(self.edge[self.slack[x]][x]), 0)==0:
                    if self.on_found_edge(self.edge[self.slack[x]][x]):
                        return True
            for b in range(self.node_cnt+1, self.n_x+1):
                if self.st[b]==b and self.S[b]==1 and my_cmp(self.lab[b], 0)==0:
                    self.expand_blossom(b)

            # print("finish end:")
            # self.print_log()
        
        return False


    #blossom algorithm
    def blossom_one_round(self, gpu_packing, ordering=1):
        ready_packing = list()
        test_required_gpu = 0
        for key in gpu_packing:
            test_required_gpu += len(gpu_packing[key])*key
        if test_required_gpu != self.required_gpu:
            print('before blossom one round: test_required_gpu != self.required_gpu', test_required_gpu, self.required_gpu)
            for key in gpu_packing:
                for packing in gpu_packing[key]:
                    print(key, [rjob.job_idx for rjob in packing.packing_jobs])

        for key in gpu_packing:
            tmp_nodes = gpu_packing[key]
            self.node_cnt = len(tmp_nodes)
            self.n_x = self.node_cnt
            self.edge = [[{"u":u, "v":v, "w":0.0} for v in range(self.node_cnt+1)] for u in range(self.node_cnt+1)]
            self.t = 0
            self.vis = [0 for _ in range(self.node_cnt+1)]
            edge_cnt = 0

            # print("-------------graph edge-----------")
            for u in range(0, self.node_cnt):
                for v in range(0, self.node_cnt):
                    job_sum = len(tmp_nodes[u].packing_jobs)+len(tmp_nodes[v].packing_jobs)
                    if u==v or job_sum>FLAGS.packing_num:
                        continue
                    # tmp_weight = tmp_nodes[u].calc_weight(tmp_nodes[v])
                    tmp_weight = tmp_nodes[u].calc_used_ratio(tmp_nodes[v], ordering)
                    if my_cmp(tmp_weight, FLAGS.weight_lbd*(job_sum/FLAGS.multi_resource))<=0:
                        continue
                    edge_cnt += 1
                    self.edge[u+1][v+1]["w"] = tmp_weight
                    # print("node %d: %d"%(u+1, tmp_nodes[u].num_gpu), [rjob.job_idx for rjob in tmp_nodes[u].packing_jobs], ";",  "node %d: %d"%(v+1, tmp_nodes[v].num_gpu), [rjob.job_idx for rjob in tmp_nodes[v].packing_jobs], ";", self.edge[u+1][v+1]["w"])
            # print("-------------graph edge end-----------")
            
            
            self.match = [0 for _ in range(self.node_cnt+1)]
            n_matches = 0
            if edge_cnt>0:
                self.pa = [0 for _ in range(self.node_cnt+1)]
                self.st = [i for i in range(self.node_cnt+1)]
                self.flower = [list() for _ in range(self.node_cnt+1)]
                self.flower_from = [[u if u==v else 0 for v in range(self.node_cnt+1)] for u in range(self.node_cnt+1)]
                w_max = max([max([self.edge[u][v]["w"] for v in range(1, self.node_cnt+1)]) for u in range(1, self.node_cnt+1)])
                # print("w_max: ", w_max)
                self.lab = [w_max for _ in range(self.node_cnt+1)]
                while self.matching():
                    n_matches += 1
            tmp_matches = 0
            for u in range(1,self.node_cnt+1):
                v = self.match[u]
                if v>0:
                    if self.match[v]==u:
                        if v<u:
                            ready_packing.append({"jobs":[v-1, u-1], "num_gpu": key, "used_ratio":self.edge[u][v]["w"]})
                            tmp_matches+=1
                    else:
                        print('matching error: ', u, v, key)
                        ready_packing.append({"jobs":[u-1], "num_gpu": key, "used_ratio": tmp_nodes[u-1].calc_self_used_ratio()}) 
                else:
                    ready_packing.append({"jobs":[u-1], "num_gpu": key, "used_ratio": tmp_nodes[u-1].calc_self_used_ratio()})
            if tmp_matches != n_matches:
                print('tmp_matches != n_matches', key, tmp_matches, n_matches)
                for u in range(1, self.node_cnt+1):
                    print(u, self.match[u])
                print('edge_weight')
                for u in range(1, self.node_cnt+1):
                    for v in range(1, self.node_cnt+1):
                        print(self.edge[u][v]['w'], end=',')
                    print('\n')

        sorted(ready_packing, key = lambda i: (1-i["used_ratio"])*i["num_gpu"])
        new_gpu_packing = dict()
        tmp_required_gpu = 0
        for packing in ready_packing:
            num_gpu = packing["num_gpu"]
            if num_gpu not in new_gpu_packing:
                new_gpu_packing[num_gpu] = list()
            if self.required_gpu<=self.cluster_gpu:
                for idx in packing["jobs"]:
                    new_gpu_packing[num_gpu].append(gpu_packing[num_gpu][idx])
                    tmp_required_gpu += num_gpu
            else:
                tmp_packing = gpu_packing[num_gpu][packing["jobs"][0]]
                if len(packing["jobs"])==2:
                    tmp_packing.add_job(gpu_packing[num_gpu][packing["jobs"][1]])
                    self.required_gpu -= num_gpu
                tmp_required_gpu += num_gpu
                new_gpu_packing[num_gpu].append(tmp_packing)
        
        if tmp_required_gpu != self.required_gpu:
            print(tmp_required_gpu, self.required_gpu)
            print('gpu_packing:')
            for key in gpu_packing.keys():
                print(key, ": ", len(gpu_packing[key]))
                for packing in gpu_packing[key]:
                    print([rjob.job_idx for rjob in packing.packing_jobs], end=';')

            print("ready_packing:", len(ready_packing))
            for rp in ready_packing:
                print(rp)

            print("new_gpu_packing:")
            for key in new_gpu_packing.keys():
                print(key, ": ", len(new_gpu_packing[key]))
                for packing in new_gpu_packing[key]:
                    print([rjob.job_idx for rjob in packing.packing_jobs], end=';')

        test_required_gpu = 0
        for key in new_gpu_packing:
            test_required_gpu += len(new_gpu_packing[key])*key
        if test_required_gpu != self.required_gpu:
            print('after blossom one round: test_required_gpu != self.required_gpu', test_required_gpu, self.required_gpu)
            for key in gpu_packing:
                for packing in new_gpu_packing[key]:
                    print(key, [rjob.job_idx for rjob in packing.packing_jobs])
        assert tmp_required_gpu == self.required_gpu

        return new_gpu_packing
                    

    def run_jobs_to_packing(self, run_jobs):
        node = list()
        for rjob in run_jobs:
            tmp_packing = _Packing(rjob)
            node.append(tmp_packing)
            # print(tmp_packing.num_gpu)
        # print("packing")
        return node

    def run(self, run_jobs_dict, cluster_gpu, ordering=1):
        self.run_jobs_dict = run_jobs_dict
        gpu_packing = dict()
        self.required_gpu = 0
        self.cluster_gpu = cluster_gpu
        for key in self.run_jobs_dict.keys():
            gpu_packing[key] = self.run_jobs_to_packing(run_jobs_dict[key])
            self.required_gpu += len(run_jobs_dict[key])*key
        # print("required gpu: ", self.required_gpu)
        for _ in range(0, FLAGS.packing_num):
            new_gpu_packing = self.blossom_one_round(gpu_packing, ordering)
            gpu_packing = new_gpu_packing
            if self.required_gpu <= self.cluster_gpu :
                break
            # print("------------packing %d round---------------" %i)
            # for key, value in gpu_packing.items():
            #     for packing in value:
            #         print([packing_job.job_idx for packing_job in packing.packing_jobs], end=':::')
            #         print('gpu', [packing_job.num_gpu for packing_job in packing.packing_jobs])
        
        return gpu_packing

Blossom_Same = _Blossom_Same()
