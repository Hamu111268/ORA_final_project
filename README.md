# ORA_final_project

## Content

* Motivation and Background
* Methodology
* Problem Definition
* Solution
* Experiment
* Reference

## Motivation and Background

In order to solve the problem of environmental pollution and road congestion, if using carpooling services, we can reduce the amount of carbon dioxide emitted by each car.

Furthermore, increasing customer (passenger) satisfaction with shorter waiting time.

Here, we are going to use reinforcement learning to simulate the carpooling problem.

## Methodology

### Reinforcement Learning

RL is known as a semi-supervised learning model in machine learning which enables an agent to learn in an interactive environment by trial and error using feedback from its own actions and experiences.

Unlike supervised learning where the feedback provided to the agent is correct set of actions for performing a task, RL uses rewards and punishments as signals for positive and negative behavior. Thus, its goal is to find a suitable action model that would maximize the total cumulative reward of the agent.

![Image text](https://github.com/Hamu111268/ORA_final_project/blob/main/img_storage/picture1.png)

### Q-Learning

Q-learning is a model-free reinforcement learning algorithm to learn the value of an action in a particular state.

In Q-learning, the experience learned by the agent is stored in the Q table, and the value in the table expresses the long-term reward value of taking specific action in a specific state. According to the Q table, the Q learning algorithm can tell the Q agent which action to choose in a specific situation to get the largest expected reward.

![Image text](https://github.com/Hamu111268/ORA_final_project/blob/main/img_storage/picture2.png)

In a certain iteration, the agent of the Q-learning algorithm is in the state $s_t$ at the time $t$ and chooses an action $a_t$ according to the strategy. It receives the reward $r_t$ from the environment and enters the new state $s_{t+1}$, and then the Q table is updated according to the equation.

### Deep Q-Learning

When the problem scale is larger, Q-table will be inefficient. Therefore, Deep Q-learning had been developed to use a neural network to approximate the Q-value function. The state is given as the input and the Q-value of all possible actions is generated as the output.

![Image text](https://github.com/Hamu111268/ORA_final_project/blob/main/img_storage/picture3.png)

## Problem Definition

We represent the map with a grid map, and there will be multiple taxis and passengers on the map.

All passengers have their own destination.

We have to assign taxis to take all passengers to their destination.

### Constraints

* The capacity of a taxi is 2, which means that the number of passengers on each taxi shouldn’t exceed 2 at any moment.
* There are no group passengers, which means that every passenger is individual.

### Environment

* In the map, a random cost is set between every pair of adjacent nodes, the random cost is the number of steps required for the taxi to go to the next node.
* One duration is one step forward for all taxis on the map that have paired passengers.
* Waiting time of each passenger is the time from start to the time when the passenger gets on the taxi.
* An episode is completed when all passengers have been sent to their respective destinations.

## Solution

### Greedy Algorithm

Every passenger will be taken by the nearest taxi in L1 distance.

### Multi-Agent Deep Q Network

Because one taxi can be assigned to take a variable number of passengers, defining action will be difficult if an agent is represented as a taxi.

Thus we represent an agent as a passenger.

Our goal is to minimize the waiting time of every passenger.

| Name   | Description |
| ------ | ----------- |
| Agent  | One passenger. |
| State  | Positions of all taxis $(c_x, c_y)$. Positions of all passengers $(p_x, p_y)$. Destination of all passengers $(d_x, d_y)$. |
| Action | An integer $a$ that represent the index of the taxi to take the passenger. |
| Reward | Waiting time of the passenger. |

## Experiment

### DQN Loss

#### Drop 0 edge

![drop 0 edge](https://github.com/Hamu111268/ORA_final_project/blob/main/img_storage/Loss_history_dqn_drop_0.png)

#### Drop 1000 edge

![drop 1000 edge](https://github.com/Hamu111268/ORA_final_project/blob/main/img_storage/Loss_history_dqn_drop_1000.png)

#### Drop 2000 edge

![drop 2000 edge](https://github.com/Hamu111268/ORA_final_project/blob/main/img_storage/Loss_history_dqn_drop_2000.png)

### Greedy v.s. DQN

|        | drop 0 edge | drop 1000 edges | drop 3000 edges |
| ------ |:-----------:|:---------------:|:---------------:|
| DQN    | 2043.5945   | 2060.556        | 2138.1075       |
| Greedy | 2281.6255   | 2333.24         | 2388.7645       |

## Reference
