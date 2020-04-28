import logging
import argparse
import random
import time
from sortedcollections import SortedDict
import numpy as np
from scipy.optimize import Bounds, minimize

PRINT_ENABLE = False
use_optimal_shrink = False

def eq(x,mems,local_mem):
    return np.dot(x, mems) - local_mem

def eq_grad(x,mems,local_mem):
    return mems

def obj_new(x, ideal_mems, percents, profiles, gradients=None, mem_gradients=None):
    r1 = 0
    r2 = 0
    for i in range(ideal_mems.shape[0]):
        r1 += ideal_mems[i]*(1-percents[i])*(x[i]*profiles[i](x[i]) - profiles[i](1))/1000
        r2 += ideal_mems[i]*(1-percents[i])*(1-x[i])*profiles[i](x[i])/1000
    return r1/r2 

def obj_grad_new(x, ideal_mems, percents, profiles, gradients, mem_gradients):
    r1 = 0
    r2 = 0
    g1 = np.empty(ideal_mems.shape)
    g2 = np.empty(ideal_mems.shape)
    for i in range(ideal_mems.shape[0]):
        r1 += ideal_mems[i]*(1-percents[i])*(x[i]*profiles[i](x[i]) - profiles[i](1))/1000
        r2 += ideal_mems[i]*(1-percents[i])*(1-x[i])*profiles[i](x[i])/1000
        g1[i] = ideal_mems[i]*(1-percents[i])*mem_gradients[i](x[i])
        g2[i] = ideal_mems[i]*(1-percents[i])*(gradients[i](x[i]) - mem_gradients[i](x[i]))
    grads = np.empty(ideal_mems.shape)
    for i in range(ideal_mems.shape[0]):
        grads[i] = (g1[i]*r2 - r1*g2[i])/r2**2 # r3 has the same gradient as r1
    return grads

def my_print(s):
    if PRINT_ENABLE:
        print(s)

class Server:
    def __init__(self, sid, L, remotemem, max_cpus, max_mem,
                 uniform_policy, min_ratio, workload_ratios, reclamation_cpus):
        self.sid = sid
        self.alloc_mem = 0
        self.min_mem_sum = 0
        self.cur_ratio = 1
        self.executing = []
        self.last_time = 0 # added for simulation
        self.next_time = 0 # optimization
        self.L = L
        self.checkin(max_cpus, max_mem, remotemem, uniform_policy,
                     min_ratio, workload_ratios, reclamation_cpus)

    def append_job(self, workload):
        self.executing.append(workload)

    def remove_job(self, workload):
        self.executing.remove(workload)

    def checkin(self, max_cpus, max_mem, use_remote, use_uniform_policy, min_ratio, workload_ratios, reclamation_cpus):
        """
        the scheduler checks in with these params.
        we return whether we have enough resources to do the checkin.
        if True, this machine will start executing jobs
        """
        # the checkin used feasible num. of cpus and mem. now initialize
        # the machine resources
        self.total_mem = max_mem
        self.total_cpus = max_cpus
        self.free_cpus = max_cpus
        self.remote_mem = use_remote
        self.use_uniform_policy = use_uniform_policy
        self.min_ratio = min_ratio
        self.workload_ratios = workload_ratios
        self.extra_cpus = reclamation_cpus*self.remote_mem 
        logging.info("Checkin Successful")

        return True
    def update_resources(self, workload, add):
        if add:
            self.free_cpus -= workload.cpu_req
            self.alloc_mem += workload.ideal_mem
            self.min_mem_sum += workload.min_mem
        else:
            self.free_cpus += workload.cpu_req
            self.alloc_mem -= workload.ideal_mem
            self.min_mem_sum -= workload.min_mem

    def finish_job(self, old_idd):
        for workload in self.executing:
            if workload.idd == old_idd:
                self.update_resources(workload, False)
                self.remove_job(workload)
                my_print("finished {} at server {}".format(workload.get_name(), self.sid))
                break

    def fill_job(self, new_workload):
        new_idd = None
        if new_workload:
            self.update_resources(new_workload, True)
            self.append_job(new_workload)
            new_idd = new_workload.idd
            my_print("started {} at server {}".format(new_workload.get_name(), self.sid))

        if self.remote_mem:
            if self.use_uniform_policy:
                ratios = self.shrink_all_uniformly(self.executing)
            else:
                ratios = self.shrink_all_proportionally(self.executing)
                if use_optimal_shrink:
                    old_ratios = ratios
                    ratios = self.shrink_optimally(self.executing,ratios, new_idd)
        else:
            assert self.alloc_mem <= self.total_mem
            ratios = None

        self.update_all_new(self.executing, new_idd, ratios)
        self.last_time = cur_time

    def set_cur_ratio(self):
        try:
            self.cur_ratio = min(1, self.total_mem / self.alloc_mem)
        except ZeroDivisionError:
            self.cur_ratio = 1

    def update_all_new(self, workloads, new_idd=None, ratios=None):
        if ratios:
            assert len(workloads) == len(ratios)
            for w, ratio in zip(workloads,ratios):
                w.update(self.L, self.sid, cur_time, self.last_time, new_idd, ratio)
        else:
            for w in workloads:
                w.update(self.L, self.sid, cur_time, self.last_time, new_idd)

    def compute_opt_ratios(self, workloads,init_ratios,new_idd):
        #ratios = init_ratios
        el_time = cur_time - self.last_time
        min_ratios = np.array([w.min_ratio for w in workloads])
        ideal_mems = np.array([w.ideal_mem for w in workloads])
        percents = np.array([(1-(w.idd==new_idd))*min(w.percent + el_time/w.profile(w.ratio), 1) for w in workloads])
        profiles = [w.profile for w in workloads]
        mem_gradients = [w.mem_gradient for w in workloads]
        gradients = [w.gradient for w in workloads]
        x0 = np.array(init_ratios)
        eq_cons = {'type': 'eq',  'fun' : eq, 'jac': eq_grad, 'args': (ideal_mems,self.total_mem)}
        bounds = Bounds(0.5, 1.0)
        res = minimize(obj_new, x0, method='SLSQP', jac=obj_grad_new, args=(ideal_mems, percents, profiles, gradients, mem_gradients), constraints=eq_cons, options={'disp': False}, bounds=bounds)
        final_ratios = res.x
        return np.round(final_ratios,3), res.fun

    def shrink_all_uniformly(self, workloads):
        total_ideal_mem = sum([w.ideal_mem for w in workloads])
        try:
            local_ratio = min(1, self.total_mem / total_ideal_mem)
        except ZeroDivisionError:
            local_ratio = 1

        assert local_ratio >= self.min_ratio
        self.set_cur_ratio()
        return [local_ratio for w in workloads]

    def shrink_all_proportionally(self, workloads):
        assert self.min_mem_sum <= self.total_mem

        total_ideal_mem = sum([w.ideal_mem for w in workloads])
        total_min_mem = sum([w.min_mem for w in workloads])
        memory_pool = total_ideal_mem - total_min_mem
        excess_mem = max(0, total_ideal_mem - self.total_mem) # Prevent containers from overgrowing
        ratios = []
        # Shrink each container
        for w in workloads:
            try:
                share_of_excess = (w.ideal_mem - w.min_mem) / memory_pool * excess_mem
            except ZeroDivisionError:
                # The pool of memory allowed to be pushed to remote storage is empty
                share_of_excess = 0
            ratio = (w.ideal_mem - share_of_excess) / w.ideal_mem
            ratios.append(ratio)
        return ratios

    def shrink_optimally(self,workloads,init_ratios, new_idd):
        total_ideal_mem = sum([w.ideal_mem for w in workloads])
        excess_mem = max(0, total_ideal_mem - self.total_mem)
        if excess_mem <= 0:
            return init_ratios
        names = [w.get_name() for w in workloads]
        if excess_mem > 0:
           ratios,_ = self.compute_opt_ratios(workloads,init_ratios,new_idd)
           ratios = ratios.tolist()
        return ratios

    def fits_remotemem(self, w, avail_far_mem):
        """ assumes the workload didn't fit normally, try to fit it with
        remote memory. we only want to determine whether the workload might
        fit, but will let the server compute its own ratio (to avoid consistency
        issues)"""
        if not self.remote_mem:
            return False
        if not self.fits_cpu(w):
            return False

        if self.use_uniform_policy:
            local_alloc_mem = self.alloc_mem + w.ideal_mem
            local_ratio = min(1, self.total_mem / local_alloc_mem)
            if local_ratio >= self.min_ratio:
                if not avail_far_mem or local_alloc_mem - self.total_mem <= avail_far_mem:
                   return True
        else:
            local_alloc_mem = self.alloc_mem + w.ideal_mem
            local_min_mem_sum = self.min_mem_sum + w.min_mem
            if local_min_mem_sum <= self.total_mem:
                if avail_far_mem is None  or local_alloc_mem - self.total_mem <= avail_far_mem:
                    return True

        return False

    def fits_normally(self, w):
        free_mem = self.total_mem - self.alloc_mem
        return self.fits_all_cpu(w) and free_mem >= w.ideal_mem

    def fits_cpu(self, w):
        return self.free_cpus >= w.cpu_req

    def fits_all_cpu(self, w):
        return (self.free_cpus + self.extra_cpus) >= w.cpu_req

class Event:
    def __init__(self,sid,wname,idd,start):
        self.sid = sid
        self.wname = wname # name to get the req for scheduling
        self.idd = idd
        self.start = start
    def event_to_workload(self,workload_ratios):
        new_workload_class = workloads.get_workload_class(self.wname)
        new_workload = new_workload_class(self.idd)
        if self.wname in workload_ratios:
            new_workload.set_min_ratio(workload_ratios[self.wname])
        return new_workload
    def get_name(self):
        return self.wname + str(self.idd)

class Schedule:
    def __init__(self):
        self.sd = SortedDict() # maintain a sorted dictionary, key is time stamp, value is (sid, idd, start/end)
    def add_event(self, timestamp, sid, wname, idd, start):
        while timestamp in self.sd:
            if timestamp == cur_time:
                print('duplicate start')
                timestamp += 0.0001
            else:
                print('duplicate end')
                timestamp -= min(0.0001, (cur_time - timestamp)/2)
        self.sd[timestamp] = Event(sid, wname, idd, start)
        return timestamp
    def delete_event(self, timestamp):
        if timestamp in self.sd: # avoid key error in redundant deletes
            del self.sd[timestamp]
    def next_event(self):
        return self.sd.popitem(0) # return (key,value) tuple
    def is_empty(self):
        return len(self.sd) == 0
    def size(self):
        return len(self.sd)
    def get_next_time(self):
        if len(self.sd) > 0:
            return self.sd.peekitem(0)[0]
        return None


def find_server_fits(servers, workload, max_far_mem, server_seq):
    # seq = list(range(len(servers)))
    # random.shuffle(seq) # random iteration
    seq = server_seq
    if not servers:
        return None
    # first try to fit the workload normally
    for i in seq:
        s = servers[i]
        if s.fits_normally(workload):
            return s
    # normal placement didn't work, are we using remote memory?
    if not servers[0].remote_mem:
        return None

    # we are using remote memory. for every server, check if we
    # can fit it using remote mem
    total_far_mem_used = sum([max(s.alloc_mem - s.total_mem,0) for s in servers])

    for i in seq:
        s = servers[i]
        others_far_mem_used = total_far_mem_used - max(0, s.alloc_mem - s.total_mem)
        if max_far_mem > 0: # has a limit
            avail_far_mem = max_far_mem - others_far_mem_used
        else: # no limits
            avail_far_mem = None
        if s.remote_mem and s.fits_remotemem(workload, avail_far_mem):
            return s

    return None

def find_new_workload(servers, Pending, max_far_mem, server_seq):
    # sequential
    tried_names = []

    while True:
        next_workload = None
        next_time = None
        next_name = None
        for name, l in Pending.items():
            if len(l) == 0 or name in tried_names:
                continue
            workload, timestamp = l[0]
            if next_name is None or timestamp < next_time:
                next_workload = workload
                next_time = timestamp
                next_name = name
        if next_workload:
            tried_names.append(next_name)
            s = find_server_fits(servers, next_workload, max_far_mem, server_seq)
            if s:
                Pending[next_name].pop(0)
                return s, next_workload
        else:
            break

    return None, None

def get_avail_far_mem(servers, s, max_far_mem):
    if max_far_mem == 0:
        return None
    others_far_mem_used = 0
    for ss in servers:
        if ss != s:
            others_far_mem_used += max(0, ss.alloc_mem - ss.total_mem)
    return max_far_mem - others_far_mem_used

def update_server_seq(default_seq,server_seq):
    server_seq = default_seq[:]
    random.shuffle(server_seq)

def schedule(servers, L, workload_ratios, max_far_mem, jobs_ts):
    Pending = dict() #key is wname, value is a list of workloads and timestamp of that name in pending
    global cur_time
    default_seq = list(range(len(servers)))
    server_seq = default_seq[:]
    far_mem_usage = []
    local_mem_usage = []
    avail_list = []

    while not L.is_empty():
        timestamp, event = L.next_event()
        cur_time = timestamp
        my_print('timestamp: {} ms'.format(cur_time))
        if event.start: # start node
            workload = event.event_to_workload(workload_ratios)
            s = None
            if not workload.wname in Pending:
                Pending[workload.wname] = [] # initialize
            if len(Pending[workload.wname]) == 0:
                s = find_server_fits(servers, workload, max_far_mem, server_seq) # only need to do this when pending is empty for the class
            if s:
                update_server_seq(default_seq,server_seq)
                s.fill_job(workload)
                jobs_ts[workload.idd]['exec'] = cur_time
            else:
                my_print("job {} can't fit".format(workload.get_name()))
                Pending[workload.wname].append((workload,cur_time))
        else: # end node
            next_time = L.get_next_time()
            old_idd = event.idd
            old_s = servers[event.sid]
            old_s.finish_job(old_idd) # finish job will update resource
            jobs_ts[old_idd]['finish'] = cur_time
            ids = [] # sid's that have been updated
            s, new_workload = find_new_workload(servers, Pending, max_far_mem, server_seq)
            while new_workload:
                next_time = L.get_next_time() # next time may change as jobs are added
                if next_time:
                    cur_time += random.uniform(0, min(100, (next_time - cur_time)*0.5)) # don't want to excde next time
                else:
                    cur_time += random.uniform(0, 100)
                s.fill_job(new_workload)
                jobs_ts[new_workload.idd]['exec'] = cur_time
                update_server_seq(default_seq,server_seq)
                ids.append(s.sid)
                s, new_workload = find_new_workload(servers, Pending, max_far_mem, server_seq)
            if not (old_s.sid in ids): # only when it has not been updated
                old_s.fill_job(None)

        my_print('')

    total_pending = 0
    for v in Pending.values():
        total_pending += len(v)
    assert total_pending == 0

    return round(cur_time)

def get_schedule(size, max_arrival, workloads, ratios, jobs_ts):
    L = Schedule()
    wid = 0
    def add_workload(name):
        nonlocal wid
        nonlocal jobs_ts
        ts = random.uniform(0, max_arrival) # uniform
        L.add_event(ts, 0, name, wid, True) # sid doesn't matter for start nodes, set when scheduled
        assert wid not in jobs_ts
        jobs_ts[wid] = {'arrival': ts, 'exec': 0, 'finish': 0}
        wid += 1

    assert len(workloads) == len(ratios)
    ratios = list(map(int, ratios))
    # this is what a ratio of 1 corresponds to
    unit = int(size / sum(ratios))
    for workload_name, ratio in zip(workloads, ratios):
        times = unit * ratio
        for _ in range(times):
            add_workload(workload_name)

    return L


def simulate(seed, mem, size, until, ratios, workload, cpus, num_servers, remotemem, workload_ratios, max_far=0, use_shrink=False, uniform=False, min_ratio=None, use_small_workload=False):
    global workloads
    if use_small_workload:
        import workloads_small as workloads
        reclamation_cpus = 1 # use 1 core for small workload (8 cpus, 32G)
    else:
        import workloads_simu as workloads
        reclamation_cpus = 3 # use 3 cores for large workload (48 cpus, 192G)

    global cur_time
    cur_time = 0 # gloabl current time
    global use_optimal_shrink
    use_optimal_shrink = use_shrink
    # min_ratio must be specified if the uniform mem policy is used
    try:
        assert (not uniform) or (uniform and min_ratio)
    except AssertionError:
        raise RuntimeError("If uniform policy is used, min_ratio must be specified")

    # Put the workload_ratio values in a dictionary with the corresponding name
    if workload_ratios:
        assert len(workload_ratios) == len(workload)
        workload_ratios = dict(zip(workload, workload_ratios))
        for k in workload_ratios.keys():
            workload_ratios[k] = workload_ratios[k]/100
    else:
        workload_ratios = dict()

    until = until * 1000  # seconds -> ms
    # Instantiate Servers
    random.seed(seed)
    jobs_ts = {}
    L = get_schedule(size, until, workload, ratios, jobs_ts)
    servers = []
    for sid in range(num_servers):
        servers.append(Server(sid, L, remotemem, cpus, mem,
                              uniform, min_ratio, workload_ratios, reclamation_cpus))
    try:
        return schedule(servers, L, workload_ratios, max_far, jobs_ts)
    except KeyboardInterrupt:
        for s in servers[:]:
            del s

