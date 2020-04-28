#!/bin/bash
python3 simulation_one_time.py $1 --num_servers 8 --cpus 7 --mem 32768 --workload_ratios 40,40,40 --remotemem --until 15 --size 600 --max_far $2 --use_shrink
python3 simulation_one_time.py $1 --num_servers 8 --cpus 7 --mem 32768 --workload_ratios 40,50,40 --remotemem --until 15 --size 600 --max_far $2 
python3 simulation_one_time.py $1 --num_servers 8 --cpus 7 --mem 32768 --workload_ratios 55,65,60 --remotemem --until 15 --size 600 --max_far $2  
python3 simulation_one_time.py $1 --num_servers 8 --cpus 7 --mem 32768 --workload_ratios 75,80,65 --remotemem --until 15 --size 600 --max_far $2 
python3 simulation_one_time.py $1 --num_servers 8 --cpus 8 --mem 32768 --until 150 --size 600 
python3 simulation_one_time.py $1 --num_servers 9 --cpus 8 --mem 32768 --until 150 --size 600 
python3 simulation_one_time.py $1 --num_servers 1 --cpus 64 --mem $((262144+$2)) --until 150 --size 600 