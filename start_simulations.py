import numpy as np
import multiprocessing as mp
from simulation_loop import get_simulate_loop
import argparse
import random
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_random', '-n', type=int,
                        help='number of randomly generated samples ' \
                        'default is 4000', default=4000)
    parser.add_argument('--limits', '-l', type=lambda s: s.split(','),
                        help='limits of m2c, format (min_limit, max_limit)', default='0.75,1.8')
    parser.add_argument('--cpu', '-c', type=int,
                        help='number of cpus required for each server, default is 48',
                        default=48)
    parser.add_argument('--mem', '-m', type=int,
                        help='memory required for each server (MB), default is 196608',
                        default=196608)
    parser.add_argument('--jps', '-j', type=int, help='number of jobs per server, default is 150', default=150)
    parser.add_argument('--filename', '-f', help='filename for final results, default is results/results_192G_48cores', default='results/results_192G_48cores')
    parser.add_argument('--simu_name', '-s', help='name of the simulation loop function, default is vary_far_mem', default='vary_far_mem')
    parser.add_argument('--use_small_workload', action='store_true', help='use small workload')
    cmdargs = parser.parse_args()
    if cmdargs.limits:
        limits = list(map(float, cmdargs.limits))
        assert len(limits) == 2
    else:
        limits = None

    global workloads
    if cmdargs.use_small_workload:
        import workloads_small as workloads # 32G
    else:
        import workloads_simu as workloads # 192G

    global jobs_per_server
    jobs_per_server = cmdargs.jps
    get_matched_pairs(cmdargs.num_random, cmdargs.cpu, cmdargs.mem, limits)
    simulate(cmdargs.simu_name, cmdargs.filename)

def get_matched_pairs(num_random, cpu, mem, limits=None):
    work_names = ['quicksort','kmeans','memaslap','linpack','spark','tf-inception']
    cpu_reqs = np.array([workloads.get_workload_class(name).cpu_req for name in work_names])
    mem_reqs = np.array([workloads.get_workload_class(name).ideal_mem for name in work_names])
    time_reqs = np.array([workloads.get_workload_class(name)(0).profile(1)/1000 for name in work_names])
    m2cs = mem_reqs/cpu_reqs
    mem_likelihoods = m2cs/np.sum(m2cs)
    cpu_likelihoods = (1/m2cs)/np.sum((1/m2cs))
    mem_likelihoods = mem_likelihoods**5/np.sum(mem_likelihoods**5)
    cpu_likelihoods = cpu_likelihoods**5/np.sum(cpu_likelihoods**5)
    cpu_intense = 0
    mem_intense = 0
    info = dict()
    shares = 100
    for i in range(num_random):
        x = i/num_random
        likelihood = (mem_likelihoods - cpu_likelihoods)*x + cpu_likelihoods
        ratios = n_ints_summing_to_v(6, shares, likelihood)
        if np.any(ratios == 0):
            num_zeros = np.sum(ratios == 0)
            ratios[ratios == 0] = 1
            ratios[np.argmax(ratios)] -= num_zeros
        cpu_pressure = np.dot(ratios, np.multiply(cpu_reqs,time_reqs))/np.sum(ratios)*jobs_per_server/(cpu)
        mem_pressure = np.dot(ratios, np.multiply(mem_reqs,time_reqs))/np.sum(ratios)*jobs_per_server/(mem)
        m2c = mem_pressure/cpu_pressure
        if m2c < 1:
            cpu_intense += 1
        else:
            mem_intense +=1
        if not m2c in info:
            info[m2c] = (ratios, mem_pressure, cpu_pressure)

    match(info, limits)

def merge_results(filename):
    cpu_count = mp.cpu_count()
    header = 'workload_ratios,job_density,large_bin,nofar,far + 192,far + 384,far + 768,far + 1536,far + 2112,far + 3072,far + 4032,nofar + 1920,nofar + 3840\n'
    with open(filename,'w') as f:
        f.write(header)
        for id in range(cpu_count):
            with open('results/results_'+str(id), 'r') as g:
                f.write(g.read())

def simulate(simu_name, filename):
    cpu_count = mp.cpu_count()
    p = mp.Pool(processes = cpu_count)
    try:
        print('starting the pool map')
        p.map(get_simulate_loop(simu_name), range(cpu_count))
        p.close()
        print('pool map complete')
    except KeyboardInterrupt:
        print('got ^C while pool mapping, terminating the pool')
        p.terminate()
        print('pool is terminated')
        return
    finally:
        print('joining pool processes')
        p.join()
        print('join complete')
    merge_results(filename)
    print('the end')

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

def in_limits(m2c, limits):
    if limits is None:
        return True
    if (m2c >= limits[0] and m2c <= limits[1]):
        return True
    return False

def sample_to_uniform(matched_pair):
    pairs = list(matched_pair.keys())
    pairs.sort(reverse=True, key=lambda x:x[1])
    max_m2c = pairs[0][1]
    bin_size = 0.1
    bins = []
    for p in pairs:
        bin_idx = int((max_m2c - p[1])/bin_size)
        if bin_idx >= len(bins):
            bins.append([p])
        else:
            bins[bin_idx].append(p)
    sample_keys = []
    num_per_bin = int(np.median([len(b) for b in bins]))
    for b in bins:
        step_size = max(1, len(b)/num_per_bin)
        idx = 0
        while idx <= len(b)-1:
            sample_keys.append(b[round(idx)])
            idx += step_size
    return sample_keys


def match(info, limits=None):
    matched_pair = dict() # key is (k1,k2), value is the change in pressure
    max_key = max(info.keys())
    search_keys = np.array(list(info.keys()))
    search_keys = search_keys[search_keys>1]
    search_keys = search_keys.tolist()

    for k in sorted(info.keys()):
        if k > 1:
            break
        if k < 1/max_key:
            continue
        idx = find_nearest(search_keys,1/k)
        best_matched_k = search_keys[idx]
        del search_keys[idx]
        matched_pair[(k,best_matched_k)] = info[best_matched_k][1]/info[k][2]

    cpu_count = mp.cpu_count()
    sample_keys = sample_to_uniform(matched_pair)
    random.shuffle(sample_keys)

    if not os.path.exists('matched'):
        os.makedirs('matched')
    if not os.path.exists('results'):
        os.makedirs('results')

    valid_list = [] 
    for (k1, k2) in sample_keys:
        density_ratio = matched_pair[(k1,k2)]
        if in_limits(k1, limits):
            valid_list.append((k1,density_ratio))
        if in_limits(k2, limits):
            valid_list.append((k2,1/density_ratio))
    num_valid = len(valid_list)
    valid_per_core =  int(num_valid/cpu_count)

    for core in range(cpu_count):
        with open('matched/matched_pair_'+str(core),'w') as f:
            for k, density_ratio in valid_list[valid_per_core*core:valid_per_core*(core+1)]:
                f.write('{},{}\n'.format(','.join(map(str, info[k][0])),int(jobs_per_server*max(1,density_ratio)+0.5)))
    print('{} matched pairs'.format(len(matched_pair)))
    print('{} valid ones'.format(num_valid))

def n_ints_summing_to_v(n, v, l):
    return np.random.multinomial(v, l)

if __name__ == '__main__':
    main()
