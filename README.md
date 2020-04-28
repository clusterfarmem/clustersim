# Cluster simulation with far memory

## Pre-requisites

python3, numpy, scipy, sortedcollections

## Paper
Please refer to our [paper](https://dl.acm.org/doi/abs/10.1145/3342195.3387522) accepted at [EUROSYS 2020](https://www.eurosys2020.org/)

## `start_simulations.py`
`start_simulations.py` is the start point from which you can run various rack scale simulations. It accepts multiple arguments, but you can generate the large scale simulation results we presented in our paper (Figure 7, 8 and 9) with the default configuration:
```
python3 start_simulations.py 
```
This would run the default large-scale simulation where the amount of far memory and additional local memory vary, the results would be written in a text file stored in results/results_192G_48cores.

### Other Arguments
Argument            | Description
--------------------------------|---------------------------------------------
--num_random, -n           | Number of randomly generated workloads.
--limits, -l       | Limits of m2c.
--cpu, -c | Number of cpu per machine.
--mem, -m              | Amount of memory per machine (unit is MB). 
--jps, -j            | Number of jobs per server.
--filename, -f           | Filename for final results.
--simu_name, -s            | Name of the simulation loop function.
--use_small_workload           | To use small workload.

## `simulation_one_time.py` and `test.sh`
`simulation_one_time.py` allows you to run single simulation with small workloads. `test.sh` contains examples usage of `simulation_one_time.py`. `test.sh` requires two parameters: seed to generate workload and amount of far memory. Here is an example usage:
```
./test.sh 2000 32768 
```
## Questions
For additional questions please contact us at cfm@lists.eecs.berkeley.edu
