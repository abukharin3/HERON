##
## '''
## Copyright (c) 2022, salesforce.com, inc.
## All rights reserved.
## SPDX-License-Identifier: BSD-3-Clause
## For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
## '''##
#code_path=outputs/codes/
#output_path=outputs/test_results/
#test_path=/export/home/apps/data/APPS/test/

#code_path=/mnt/t-qingru/RLHF/outputs/codet5-rl/codes_temp0.1_seq10/
#output_path=/mnt/t-qingru/RLHF/outputs/codet5-rl/results/codes_temp0.1_seq10/
#test_path=/mnt/t-qingru/RLHF/APPS/test/

# First cmd argument is code_path, 2nd is output_path, 3rd is test_path
code_path=$1
output_path=$2
test_path=$3

example_tests=$4 # 0: run hidden unit tests; 1: run example unit tests
start=0
end=5000
threads=30

if [ ! -d $output_path ] 
then
    echo "Directory DOES NOT exists." 
    mkdir $output_path
fi

index=0
for (( i=$start;i<$end;i++ )) ; do 
    echo 'testing sample index #' ${i}
    ((index++))   
    (
    python test_one_solution.py \
        --code_path ${code_path} \
        --output_path ${output_path} \
        --test_path $test_path \
        --example_tests $example_tests \
        --i $i 
    ) &        
    if (( $index % $threads == 0 )); then wait; fi 
done 

wait 
