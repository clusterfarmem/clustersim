from simulation import simulate
import random
import numpy as np
import pickle

def simulate_loop_rep(id, num_trials=25):
    '''
    Simulate_loop for 1) local 2) extra local 3)less local + far (to compare with 1) 4) local + far (to compare with 2)
    '''
    jobs = dict() # key is workload ratios, value is job density
    with open('matched/matched_pair_'+str(id),'r') as r:
        line = r.readline()
        while line:
            s = line.strip().split(',')
            assert len(s) == 7
            s = list(map(int,s))
            jobs[tuple(s[0:6])] = s[6]
            line = r.readline()

    print('id {} starts with {} jobs, each with {} trials'.format(id, len(jobs), num_trials))

    with open('results/results_'+str(id), 'w') as f: # empty/create file
        pass
    workload = ['quicksort', 'kmeans', 'memaslap', 'linpack', 'spark', 'tf-inception']
    #workload_ratios = [55, 75, 70, 75, 90, 50]
    workload_ratios = [55, 65, 60, 75, 90, 50]
    min_ratios = [40, 40, 40, 50, 50, 50]
    mem = 32768
    until = 10 #1500
    cpus = 8
    cpus_far = 7
    num_servers = 8
    num_extra_servers = 10#int(num_servers*5/4)
    max_far = (num_extra_servers - num_servers)*mem
    mem_less = mem - max_far/num_servers

    for ratios, job_density in jobs.items():
        ts = np.zeros((4,))
        best_ratio = -1
        for i in range(num_trials):
            seed = random.randint(0,100000)
            print(seed)
            remotemem = False
            t0 = simulate(seed, mem, num_servers*job_density, until, ratios, workload, cpus, num_servers, remotemem, workload_ratios, use_small_workload=True)
            t1 = simulate(seed, mem, num_servers*job_density, until, ratios, workload, cpus, num_extra_servers, remotemem, workload_ratios, use_small_workload=True)
            remotemem = True
            t2 = simulate(seed, mem_less, num_servers*job_density, until, ratios, workload, cpus_far, num_servers, remotemem, workload_ratios, max_far, use_small_workload=True)
            t3 = simulate(seed, mem, num_servers*job_density, until, ratios, workload, cpus_far, num_servers, remotemem, workload_ratios, max_far, use_small_workload=True)
            ts[0] += t0
            ts[1] += t1
            ts[2] += t2
            ts[3] += t3
            if t1/t3 > best_ratio:
                best_ratio = t1/t3
                best_seed = seed
        ts = np.rint(ts/num_trials).astype(int)
        with open('results/results_'+str(id), 'a') as f:
            r_str = ':'.join(map(str,ratios))
            t_str = ','.join(map(str,ts))
            f.write('{},{},{},{}\n'.format(r_str, job_density, t_str, best_seed))

def simulate_loop_vary_s(id, num_trials=25):
    '''
    Simulate_loop for various number of server ( 40 + 0G, 39 + 32G, 38 + 64G, etc)
    '''
    jobs = dict() # key is workload ratios, value is job density
    with open('matched/matched_pair_'+str(id),'r') as r:
        line = r.readline()
        while line:
            s = line.strip().split(',')
            assert len(s) == 7
            s = list(map(int,s))
            jobs[tuple(s[0:6])] = s[6]
            line = r.readline()

    print('id {} starts with {} jobs, each with {} trials'.format(id, len(jobs), num_trials))

    with open('results/results_'+str(id), 'w') as f: # empty/create file
        pass
    workload = ['quicksort', 'kmeans', 'memaslap', 'linpack', 'spark', 'tf-inception']
    workload_ratios = [55, 65, 60, 75, 90, 50]
    mem = 32768
    until = 10#1500
    cpus_far = 7
    num_servers = 40
    seeds = [random.randint(0,100000) for i in range(num_trials)]
    vary_range = 5#int(num_servers/2)
    remotemem = True
    extra_mem_ratio = 1

    for ratios, job_density in jobs.items():
        # compute optimal
        ts_optimal = 0
        print('compute optimal')
        for i in range(num_trials):
            seed = seeds[i]
            ts_optimal += simulate(seed, mem*num_servers, num_servers*job_density, until, ratios, workload, (cpus_far+1)*num_servers, 1, False, workload_ratios, use_small_workload=True)
        ts_optimal = int(ts_optimal/num_trials)

        ts = np.zeros((vary_range,))
        for num_mem_server in range(vary_range):
            max_far = max(num_mem_server*mem, 1)*extra_mem_ratio # 0 would be no limits
            num_local_server = num_servers - num_mem_server
            print(num_mem_server)

            for i in range(num_trials):
                seed = seeds[i]
                ts[num_mem_server] += simulate(seed, mem, num_servers*job_density, until, ratios, workload, cpus_far, num_local_server, remotemem, workload_ratios, max_far, use_small_workload=True)

        ts = np.rint(ts/num_trials).astype(int)

        with open('results/results_'+str(id), 'a') as f:
            r_str = ':'.join(map(str,ratios))
            t_str = ','.join(map(str,ts))
            f.write('{},{},{},{}\n'.format(r_str, job_density, ts_optimal, t_str))

def simulate_loop_fixed_far_mem(id, num_trials=25):
    '''
    Simulate_loop for fixed far mem ( 512 GB far memory, then reduce local memory gradually 1GB by 1GB)
    '''
    jobs = dict() # key is workload ratios, value is job density
    with open('matched/matched_pair_'+str(id),'r') as r:
        line = r.readline()
        while line:
            s = line.strip().split(',')
            assert len(s) == 7
            s = list(map(int,s))
            jobs[tuple(s[0:6])] = s[6]
            line = r.readline()

    print('id {} starts with {} jobs, each with {} trials'.format(id, len(jobs), num_trials))

    with open('results/results_'+str(id), 'w') as f: # empty/create file
        pass
    workload = ['quicksort', 'kmeans', 'memaslap', 'linpack', 'spark', 'tf-inception']
    workload_ratios = [55, 65, 60, 75, 90, 50]
    mem = 32768
    until = 10#1500
    cpus_far = 7
    num_servers = 40
    seeds = [random.randint(0,100000) for i in range(num_trials)]
    vary_range = round(512/num_servers)#int(num_servers/2)
    remotemem = True
    extra_mem_ratio = 1

    for ratios, job_density in jobs.items():
        ts = np.zeros((vary_range,))
        for idx in range(vary_range):
            num_local_servers = num_servers
            print(idx)
            if idx == 0:
                cpus = cpus_far + 1
                remotemem = False
                max_far = 0
                num_local_servers = 40
                mem = 32768
            else:
                cpus = cpus_far
                remotemem = True
                max_far = 524288
                num_local_servers = 39
                mem = 1024*(32 - idx - 1)

            for i in range(num_trials):
                seed = seeds[i]
                ts[idx] += simulate(seed, mem, num_servers*job_density, until, ratios, workload, cpus, num_local_servers, remotemem, workload_ratios, max_far, use_small_workload=True)

        ts = np.rint(ts/num_trials).astype(int)

        with open('results/results_'+str(id), 'a') as f:
            r_str = ':'.join(map(str,ratios))
            t_str = ','.join(map(str,ts))
            f.write('{},{},{}\n'.format(r_str, job_density, t_str))
        #idx += 1

def simulate_loop_vary_far_mem(id, num_trials=10):
    '''
    Simulate_loop for varied far mem ( fixed local memory, increase far memory by power of 2)
    '''
    jobs = dict() # key is workload ratios, value is job density
    with open('matched/matched_pair_'+str(id),'r') as r:
        line = r.readline()
        while line:
            s = line.strip().split(',')
            assert len(s) == 7
            s = list(map(int,s))
            jobs[tuple(s[0:6])] = s[6]
            line = r.readline()

    print('id {} starts with {} jobs, each with {} trials'.format(id, len(jobs), num_trials))

    with open('results/results_'+str(id), 'w') as f: # empty/create file
        pass
    workload = ['quicksort', 'kmeans', 'memaslap', 'linpack', 'spark', 'tf-inception']
    workload_ratios = [55, 65, 60, 75, 90, 50]
    mem = 196608
    until = 10
    cpus_far = 45
    num_servers = 40
    seeds = [random.randint(0,100000) for i in range(num_trials)]
    remotemem = True
    extra_mem_ratio = 1

    for ratios, job_density in jobs.items():
        print('compute largebin')
        # largebin has the number of cpus and memory that nofar has. largebin tput / nofar tput
        # is a metric of how much resource fragmentation explains the performance of nofar
        cpus_largebin = (cpus_far + 3) * num_servers
        mem_largebin = mem * num_servers
        ts_largebin = 0

        # large bin case
        for seed in seeds:
            mkspan = simulate(seed, mem_largebin, num_servers*job_density, until,
                    ratios, workload, cpus_largebin, 1, False, workload_ratios)
            ts_largebin += mkspan
        ts_largebin = int(ts_largebin/num_trials)

        # extra far memory case
        possible_far_mem = [0, 192,384,768,1536,2112,3072,4032]
        ts = np.zeros((len(possible_far_mem),))
        use_shrink = False
        for f, far_mem in enumerate(possible_far_mem):
            num_local_servers = num_servers
            print('far memory {} GB'.format(far_mem))
            if f == 0:
                cpus = cpus_far + 3
                remotemem = False
                max_far = 0
                num_local_servers = 40
            else:
                cpus = cpus_far
                remotemem = True
                max_far = far_mem * 1024
                num_local_servers = 39
                use_shrink = True

            for i in range(num_trials):
                mkspan = simulate(seeds[i], mem, num_servers*job_density, until, ratios, workload,
                            cpus, num_local_servers, remotemem, workload_ratios, max_far, use_shrink=use_shrink)
                ts[f] += mkspan

        ts = np.rint(ts/num_trials).astype(int)

        # extra local memory case
        extra_mem_per_node = [48,96]
        extra_mem_ts = np.zeros((len(extra_mem_per_node),))
        cpus_nofar = cpus_far + 3
        for e, extra_mem in enumerate(extra_mem_per_node):
            print("extra_mem {} GB".format(extra_mem))
            mem_nofar = mem + (extra_mem * 1024)
            for i in range(num_trials):
                extra_mem_ts[e] += simulate(seeds[i], mem_nofar, num_servers*job_density, until,
                        ratios, workload, cpus_nofar, 40, False, workload_ratios)
        extra_mem_ts = np.rint(extra_mem_ts/num_trials).astype(int)

        with open('results/results_'+str(id), 'a') as f:
            r_str = ':'.join(map(str,ratios))
            t_str = ','.join(map(str,ts))
            extralocal_str = ','.join(map(str, extra_mem_ts))
            f.write('{},{},{},{},{}\n'.format(r_str, job_density, ts_largebin, t_str, extralocal_str))

def simulate_loop_portional(id, num_trials=25):
    '''
    Simulate_loop for portioanl case ( 32G + x local vs 32G + x*num_server*extra_mem_ratio far)
    '''
    jobs = dict() # key is workload ratios, value is job density
    with open('matched/matched_pair_'+str(id),'r') as r:
        line = r.readline()
        while line:
            s = line.strip().split(',')
            assert len(s) == 7
            s = list(map(int,s))
            jobs[tuple(s[0:6])] = s[6]
            line = r.readline()

    print('id {} starts with {} jobs, each with {} trials'.format(id, len(jobs), num_trials))

    with open('results/results_'+str(id), 'w') as f: # empty/create file
        pass
    workload = ['quicksort', 'kmeans', 'memaslap', 'linpack', 'spark', 'tf-inception']
    workload_ratios = [55, 65, 60, 75, 90, 50]
    mem = 32768
    until = 10#1500
    cpus_far = 7
    num_servers = 40
    seeds = [random.randint(0,100000) for i in range(num_trials)]
    extra_mem_ratio = 2
    num_configs = 12

    for ratios, job_density in jobs.items():
        ts = np.zeros((2*num_configs,))
        for idx in range(2*num_configs):
            print(idx)
            if idx <= num_configs - 1: # no far
                cpus = cpus_far + 1
                remotemem = False
                max_far = 0
                num_local_servers = 40
                mem = 1024 * (32 + idx)
            else:
                if idx <= 2*num_configs -1: # far
                    cpus = cpus_far
                    remotemem = True
                    max_far = 1024 * (idx - num_configs) * num_servers * extra_mem_ratio + 1
                    num_local_servers = 40
                    mem = 1024*32
                else:
                    cpus = (cpus_far + 1) * num_servers
                    remotemem = False
                    max_far = 0
                    num_local_servers = 1
                    mem = (32 + extra_mem_ratio * (idx-2*num_configs)) * num_servers * 1024

            for i in range(num_trials):
                seed = seeds[i]
                ts[idx] += simulate(seed, mem, num_servers*job_density, until, ratios, workload, cpus, num_local_servers, remotemem, workload_ratios, max_far, use_small_workload=True)

        ts = np.rint(ts/num_trials).astype(int)

        with open('results/results_'+str(id), 'a') as f:
            r_str = ':'.join(map(str,ratios))
            t_str = ','.join(map(str,ts))
            f.write('{},{},{}\n'.format(r_str, job_density, t_str))
        #idx += 1

def get_simulate_loop(ftype):
    return {'rep': simulate_loop_rep,
            'vary_s': simulate_loop_vary_s,
            'fixed_far_mem': simulate_loop_fixed_far_mem,
            'vary_far_mem': simulate_loop_vary_far_mem,
            'portional': simulate_loop_portional}[ftype]
