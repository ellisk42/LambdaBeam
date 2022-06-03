#!/bin/bash

# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

data_dir=$HOME/xlambda-data/deepcoder

# TODO(kshi): I changed this script for easier testing. When everything works,
# we should revisit all the parameters.
tout=600
maxw=10
maxne=5
maxni=3
skipprob=0.75
num_proc=1
out_dir=$data_dir/t-${tout}-maxw-${maxw}-maxne-${maxne}-maxni-${maxni}-skipprob-${skipprob}

if [ ! -e $out_dir ];
then
    mkdir -p $out_dir
fi

echo 'Generating validation'
valid_file=$out_dir/valid-tasks.pkl

python3 -m crossbeam.datasets.bottom_up_data_generation \
    --domain=deepcoder \
    --output_file=$valid_file \
    --data_gen_seed=1 \
    --data_gen_timeout=$tout \
    --num_tasks=100 \
    --num_searches=1 \
    --min_task_weight=3 \
    --max_task_weight=$maxw \
    --min_num_examples=2 \
    --max_num_examples=$maxne \
    --min_num_inputs=1 \
    --max_num_inputs=$maxni \
    --skip_probability=$skipprob \
    --num_datagen_proc=$num_proc \
    --verbose=False

exit  # TODO(kshi): temporary
echo 'Generating training'

training_file=$out_dir/train-tasks.pkl

python3 -m crossbeam.datasets.bottom_up_data_generation \
    --domain=deepcoder \
    --output_file=$training_file \
    --data_gen_seed=0 \
    --data_gen_timeout=$tout \
    --num_tasks=1000 \
    --num_searches=10000 \
    --min_task_weight=3 \
    --max_task_weight=$maxw \
    --min_num_examples=2 \
    --max_num_examples=$maxne \
    --min_num_inputs=1 \
    --max_num_inputs=$maxni \
    --skip_probability=$skipprob \
    --num_datagen_proc=$num_proc \
    --verbose=False
