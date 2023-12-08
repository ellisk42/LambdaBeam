# LambdaBeam: Neural Program Search with Higher-Order Functions and Lambdas

This repository contains the source code associated with the paper published at
  NeurIPS'23 ([OpenReview](https://openreview.net/forum?id=qVMPXrX4FR)):

> Kensen Shi, Hanjun Dai, Wen-Ding Li, Kevin Ellis, and Charles Sutton.
> **LambdaBeam: Neural Program Search with Higher-Order Functions and Lambdas.**
> Conference on Neural Information Processing Systems (NeurIPS), 2023.

In this research project, we design, train, and evaluate a neural program
synthesizer that can construct arbitrary lambda functions for use within
higher-order functions to perform looping computations.

To cite this work, you can use the following BibTeX entry:
```
@inproceedings{shi2023lambdabeam,
    title={{LambdaBeam}: Neural Program Search with Higher-Order Functions and Lambdas},
    author={Kensen Shi and Hanjun Dai and Wen-Ding Li and Kevin Ellis and Charles Sutton},
    booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
    year={2023},
    url={https://openreview.net/forum?id=qVMPXrX4FR},
}
```

This repository was developed as a fork of
[CrossBeam](https://github.com/google-research/lambdabeam). As a result, some
files are carried over without playing a role in the LambdaBeam system.

## Setup

To install dependencies, run the following from the root `lambdabeam/`
directory:

```
pip3 install -e .
```

It may help to install [PyTorch](https://pytorch.org/get-started/locally/) and
`torch-scatter` with CUDA, for example with the following commands:

```
pip3 install torch==1.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip3 install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
```

## Run tests

Run `pytest` in this directory.

## View the test datasets

The 100 handwritten and 100 synthetic test tasks are all in
`lambdabeam/data/deepcoder/deepcoder_tasks.py`.

## Run the trained LambdaBeam model on test datasets

The `run_lambdabeam.sh` script runs the trained LambdaBeam model on the
handwritten and synthetic test tasks, saving results as JSON files in the
`results/` directory.

Even with a fixed seed, the actual results will vary based on your computing
setup (its speed and potentially the versions of dependencies) and randomness
due to stopping the search after the time limit is reached, which affects
random decisions for subsequent tasks.

## View the raw results we collected for NeurIPS'23

For reference, the experimental results we collected for the NeurIPS'23 paper
can be found in `results/neurips23`.

## Generate the training dataset

We do not include the training dataset in this repo, but if you wish to
generate training data, refer to the files `launch_xm_deepcoder_gen.py`,
`make_shards.py`, and `filter_trained_data.py` in the `lambdabeam/datasets`
directory, although they may need to be edited according to your setup.

## Train the model

The trained LambdaBeam model is already included in this repo as
`lambdabeam-model.ckpt`. If you wish to re-train the model, first generate
training data and then refer to `lambdabeam/experiment/launch_xm_train.sh`
which may need to be edited according to your setup.

## Disclaimer

This is not an official Google product.
