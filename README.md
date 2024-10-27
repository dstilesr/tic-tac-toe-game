# Tic Tac Toe

## Contents

- [Overview](#overview)
- [Usage](#usage)
  - [Setup](#setup)
  - [Training Agents](#training-agents)
  - [Play Against Agent](#play-against-agent)

## Overview
This repository contains some code to train and play with simple agents that play tic-tac-toe.
These agents can be trained with tabular TD-learning methods such as Q-Learning and Expected Sarsa.
Once you have trained some agents, you can play against them yourself and see how they do!

## Usage

### Setup
In order to use the project, first create a Python virtual environment with your preferred tool. Next,
you can install the libraries required to run the project by running `pip install -r requirements.txt`, or
`task install-dependencies`.

Next, you can run unit tests to ensure everything is working properly with `task unit-tests`. Then you should
be ready to go!

### Training Agents
To train agents, you can run the following in the terminal:
```shell
# Train an agent with Q-Learning against an opponent that plays randomly for 100k games
python -m tic_tac_toe.training \
       --run_name=test-01 \
       --total_episodes=100000 \
       --agent_type=q_learn \
       --opponent_type=random

# Train an agent with Q-Learning against a second agent that also learns with experience! (250K games)
python -m tic_tac_toe.training \
       --run_name=test-02 \
       --total_episodes=250000 \
       --agent_type=q_learn \
       --opponent_type=q_learn \
       --td_settings_file="configs/td-learn-cfgs/td-learn-cfg.json"
       
# Train an agent with Expected Sarsa against a learning opponent, starting from the values learned in a previous run!
python -m tic_tac_toe.training \
       --run_name=test-04 \
       --total_episodes=275000 \
       --agent_type=expected_sarsa \
       --opponent_type=q_learn \
       --td_settings_file="configs/td-learn-cfgs/td-learn-cfg-2.json" \
       --policy_file="outputs/test-01/policy.json"
```
When you run an agent training task, the learned Q-values and a summary of the training run will be stored
in the `outputs/{run name}` folder.

## Play Against Agent
Once you have trained an agent, you can play against it on the terminal by running:
```shell
# Play against randomly playing agent
python -m tic-tac-toe play --opponent_type=random

# Play against trained agent
python -m tic-tac-toe play --opponent_type=q_learn --policy_file="outputs/test-01/policy.json"
```
The cells on the 'board' are numbered as follows:
```
0|1|2
-+-+-
3|4|5
-+-+-
6|7|8
```
To make a move, input the number of the cell you want to claim and hit enter.

You can see more about the available options and tasks by running
```shell
python -m tic_tac_toe --help

python -m tic_tac_toe.training --help
```

[Back to top.](#tic-tac-toe)
