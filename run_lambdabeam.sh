#!/bin/bash

# Copyright 2023 Google LLC
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

timeout=600
seed=0
results_dir=results
mkdir -p ${results_dir}

# Run LambdaBeam on handwritten tasks.
restarts_timeout=6
python3 -m lambdabeam.experiment.run_lambdabeam \
    --config=eval_config.py \
    --config.timeout=${timeout} \
    --config.restarts_timeout=${restarts_timeout} \
    --config.seed=${seed} \
    --config.json_results_file=${results_dir}/results.handwritten.timeout-${timeout}-${restarts_timeout}.json \
    --config.load_model=lambdabeam-model.ckpt \
    --config.synthetic_test_tasks=False

# Run LambdaBeam on synthetic tasks.
restarts_timeout=30
python3 -m lambdabeam.experiment.run_lambdabeam \
    --config=eval_config.py \
    --config.timeout=${timeout} \
    --config.restarts_timeout=${restarts_timeout} \
    --config.seed=${seed} \
    --config.json_results_file=${results_dir}/results.synthetic.timeout-${timeout}-${restarts_timeout}.json \
    --config.load_model=lambdabeam-model.ckpt \
    --config.synthetic_test_tasks=True

