# APOLLO-MILP for COPT

This repository contains COPT version of APOLLO-MILP, which relies the following repositories:

- [APOLLO](https://github.com/MIRALab-USTC/Apollo-MILP) for `Apollo.py`, where we replace Gurobi with COPT.
- [Predict-and-Search_MILP_method](https://github.com/sribdcn/Predict-and-Search_MILP_method) for `GCN.py`, `trainPredictModel.py`, `helper.py` and `copt.py`, which is also used in [APOLLO](https://github.com/MIRALab-USTC/Apollo-MILP) and we replace Gurobi with COPT.
- [learn2branch](https://github.com/ds4dm/learn2branch) for `01_generate_instances.py` and `utilities.py`, which is used to generate `CA` and `SC` instances for training and testing.

## Requirements

You will need [COPT](https://www.shanshu.ai/) and [uv](https://docs.astral.sh/uv/) to set up the environment.

## Setup

To set up the environment, you can simply run:

```bash
uv sync
```

## Usage

To evaluate the performance of APOLLO-MILP for COPT on `CA` and `SC` instances, we need 4 steps:

- Step 1: Generate `CA` and `SC` instances.
- Step 2: Use COPT to solve the generated instances (train and valid) and save the results.
- Step 3: Train the GCN model using the dataset generated in Step 2.
- Step 4: Use the trained GCN model to guide COPT to solve the test instances.

### Step 1: Generate CA and SC instances

To generate `CA` and `SC` instances, we can use the `01_generate_instances.py`:

```bash
python 01_generate_instances.py -p problem -s seed -t thread_num
```

where `problem` can be `ca` or `sc`, `seed` is an integer for random seed, and `thread_num` is the number of threads to use. The generated instances will be saved in the `data` folder.

### Step 2: Use COPT to solve the generated instances (train and valid) and save the results

To use COPT to solve the generated instances and save the results, we can use the `copt.py`:

```bash
python copt.py --prob problem --nWorkers thread_num
```

where `problem` can be `ca` or `sc`, and `thread_num` is the number of threads to use. There are other arguments that you can find in the code. The results will be saved in the `dataset` folder.

### Step 3: Train the GCN model using the dataset generated in Step 2

To train the GCN model using the dataset generated in Step 2, we can use the `trainPredictModel.py`:

```bash
python trainPredictModel.py
```

You will need to modify TaskName in the code to specify the problem type, as well as training parameters. The trained model will be saved in the `pretrain` folder and the training log will be saved in the `train_logs` folder.

If you want to plot the training and validation curves, you can use the `plotLoss.py`:

```bash
python plotLoss.py
```

### Step 4: Use the trained GCN model to guide COPT to solve the test instances

To use the trained GCN model to guide COPT to solve the test instances, we can use the `Apollo.py`:

```bash
python Apollo.py -p problem -g gpu_id
```

where `problem` can be `ca` or `sc`, and `gpu_id` is the GPU ID to use. There are other arguments that you can find in the code. The results will be saved in the `logs` folder where each instance has a separate sub-folder. The sub-folder contains solver log and reduced problem in each iteration and the log for the last iteration is the final result.
