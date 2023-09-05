##
## Copyright (c) 2022, salesforce.com, inc.
## All rights reserved.
## SPDX-License-Identifier: BSD-3-Clause
## For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
##
#model_path=/mnt/t-qingru/RLHF/models/codet5-large-ntp-py
#tokenizer_path=/mnt/t-qingru/RLHF/models/codet5-large-ntp-py/
#test_path=/mnt/t-qingru/RLHF/APPS/test/
#
#start=0
#end=5000
#num_seqs_per_iter=50
#num_seqs=1000
#temp=0.6
#
#output_path=/mnt/t-qingru/RLHF/outputs/codes/
#
#CUDA_VISIBLE_DEVICES=0 python generate.py \
#    --model_path $model_path \
#    --tokenizer_path $tokenizer_path \
#    --test_path $test_path \
#    --output_path $output_path \
#    -s $start -e $end \
#    --num_seqs $num_seqs --num_seqs_per_iter $num_seqs_per_iter \
#    --temperature $temp \

model_path=/mnt/t-qingru/RLHF/models/codet5-large-ntp-py
tokenizer_path=/mnt/t-qingru/RLHF/models/codet5-large-ntp-py/
test_path=/mnt/t-qingru/RLHF/APPS/test/

start=0
end=5000
num_seqs_per_iter=50
num_seqs=1000
temp=0.6

output_path=/mnt/t-qingru/RLHF/outputs/codes/

CUDA_VISIBLE_DEVICES=0 python generate.py \
    --model_path /mnt/t-qingru/RLHF/models/codet5-large-ntp-py \
    --tokenizer_path /mnt/t-qingru/RLHF/models/codet5-large-ntp-py/ \
    --test_path /mnt/t-qingru/RLHF/APPS/test/ \
    --output_path /mnt/t-qingru/RLHF/outputs/codes/ \
    -s 0 -e 5000 \
    --num_seqs 1000 --num_seqs_per_iter 50 \
    --temperature 0.6 \