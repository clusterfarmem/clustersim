import argparse
from simulation import simulate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('seed', type=int)
    parser.add_argument('--remotemem', '-r', action='store_true',
                        help='enable remote memory')
    parser.add_argument('--num_servers','-n', type=int, help='number of servers to simulate', required=True)
    parser.add_argument('--max_far', '-s', type=int, default=0,
                        help='max size of far memory, default=0 (unlimited)')
    parser.add_argument('--cpus', '-c', type=int,
                        help='number of cpus required for each server',
                        required=True)
    parser.add_argument('--mem', '-m', type=int,
                        help='memory required for each server (MB)',
                        required=True)
    parser.add_argument('--size', type=int,
                        help='size of workload (num of tasks) ' \
                        'default=100', default=100)
    parser.add_argument('--workload', type=lambda s: s.split(','),
                        help='tasks that comprise the workload ' \
                        'default=quicksort,kmeans,memaslap',
                        default='quicksort,kmeans,memaslap')
    parser.add_argument('--ratios', type=lambda s: s.split(':'),
                        help='ratios of tasks in workload, default=2:2:1',
                        default="2:2:1")
    parser.add_argument('--until', type=int,
                        help='max arrival time in minutes default=20',
                        default=15)
    parser.add_argument('--uniform', action='store_true',
                        help='use uniform memory policy')
    parser.add_argument('--min_ratio', type=float,
                        help='smallest allowable memory ratio')
    parser.add_argument('--workload_ratios', type= lambda s: s.split(','),default="50,50,50",
                        help='ratios for each workload')
    parser.add_argument('--use_shrink', action='store_true', help='use optimization based shrinking')

    cmdargs = parser.parse_args()
    makespan = simulate(cmdargs.seed, cmdargs.mem, cmdargs.size, cmdargs.until, cmdargs.ratios, cmdargs.workload, cmdargs.cpus, cmdargs.num_servers, cmdargs.remotemem, list(map(float,cmdargs.workload_ratios)), max_far=cmdargs.max_far, use_shrink=cmdargs.use_shrink, uniform=cmdargs.uniform, min_ratio=cmdargs.min_ratio, use_small_workload=True)
    print('Makespan is {} ms'.format(makespan))

if __name__ == '__main__':
    main()       