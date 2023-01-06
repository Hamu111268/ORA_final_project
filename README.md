# ORA_final_project

## Content

* Motivation and Background
* Methodology
* Problem Definition
* Solution
* Experiment
* Reference

## Motivation and Background

Nowadays, environmental pollution and road congestion problems are getting increasingly worse, especially in densely populated areas.
If using carpooling services, we can reduce the amount of carbon dioxide emitted by each car.
Furthermore, cutting down the cost of travel for each passenger and increasing passenger satisfaction with shorter waiting time are also the advantages of carpooling services.

Here, we are going to use Reinforcement Learning to simulate the carpooling problem.

Our carpooling problem is not going to address problem of scheduling a fleet of taxis, but provide user(passenger) a option to choose a taxi which can let user has the shortest waiting time.

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

### Dijkstra Algorithm

Dijkstra algorithm is a greedy algorithm that can find the lengths of shortest path in a graph.

The input is the graph and the source node, and the distance of every other node is marked as infinity.

Below is the pseudocode:

* `dist[v]` is the length from source node to node `v`

```
function Dijkstra(Graph, source):
    for each vertex v in Graph.Vertices:
        dist[v] ← INFINITY
        prev[v] ← UNDEFINED
        add v to Q
    dist[source] ← 0

    while Q is not empty:
        u ← vertex in Q with min dist[u]
        remove u from Q

        for each neighbor v of u still in Q:
            alt ← dist[u] + Graph.Edges(u, v)
            if alt < dist[v]:
                dist[v] ← alt
                prev[v] ← u

    return dist[], prev[]
```

This algorithm always choose the node with minimum distance from `Q`, and update the distance of its neighbor nodes.

Other than `dist`, the code above also return another array `prev`.

`prev[v]` will be the previous node of `v` on the shortest path from source to `v`.

## Problem Definition

* Objective：Minimize the average duration of all episodes.

    - One duration is one step forward for all taxis on the map that have paired passengers.
    - An episode is completed when all passengers have been sent to their respective destinations.
    
* Setting up a map existing multiple taxis and passengers, which is called a grid map.

    - The map size is 100 x 100, and the coordinates of the lower left and upper right position is (0, 0) and (99, 99) respectively.

    - The number of passengers is 40, this number is fixed in every episode.

    - The number of taxi is 20, this number is fixed in every episode.

    - In the map, a random cost is set between every pair of adjacent nodes, the random cost is the number of steps required for the taxi to go to the next node.
        ```python
        """
        Set the weight of edges in the grid map with random value
        """
        def init_map_cost(self):
            for row in range(self.size[0]):
                for col in range(self.size[1]):
                    p = (row, col)

                    p_down = (row + 1, col)
                    if self.is_valid(p_down):
                        self.map_cost[(p_down, p)] = self.map_cost[(p, p_down)] = random.randint(1, 9)

                    p_right = (row, col + 1)
                    if self.is_valid(p_right):
                        self.map_cost[(p_right, p)] = self.map_cost[(p, p_right)] = random.randint(1, 9)
        ```

* We measure the duration time in discret time.
 
* Every passenger is represented as two points, pick-up position and destination.

* The capacity of taxi is 2, which means that the number of passengers on each taxi shouldn’t exceed 2 at any moment.

* There are no group passengers, which means that every passenger is individual.
* Drop by：First of all, the first passenger decides the destination first, and then a taxi will come to pick him up. At this time, if there is a second passenger who also wants to take this taxi, the system will first judge whether it is on the way (the place surrounded by the orange frame) , if there is on the way, it will pick up the second passenger.
<img width="1201" alt="截圖 2023-01-06 23 34 29" src="https://user-images.githubusercontent.com/122088028/211044853-2a7b76c2-91aa-4937-a2de-270737d8d9a1.png">



## Solution

### Greedy Algorithm

Every passenger will be taken by the nearest taxi in L1 distance.

### Multi-Agent Deep Q Network

Because one taxi can be assigned to take a variable number of passengers, defining action will be difficult if an agent is represented as a taxi.

Thus we represent an agent as a passenger.

Our goal is to minimize the waiting time of every passenger.

#### Agent

* Passengers

#### State

* $(c_x, c_y)$, positions of all taxis.
* $(p_x, p_y)$, pick-up point of all passengers.
* $(d_x, d_y)$, destination of all passengers.

#### Action

* An 1-D array $a$ that represent the index of the taxi to take the passenger.

For example:

Assume the number of passengers is 5, `[0, 3, 1, 2, 3]` maybe one possible action.

Below is the table that explains what `[0, 3, 1, 2, 3]` means.

| Passenger index | Assigned Taxi index |
|:---------------:|:-------------------:|
| 0               | 0                   |
| 1               | 3                   |
| 2               | 1                   |
| 3               | 2                   |
| 4               | 3                   |

#### Reward

* Waiting time of every passenger.

#### Transition

The position of taxis at the end of this episode will effect the duration of next episode. 

Thus we only set the passengers' positions and destinations with random numbers.

The randomly generated state will be the $s_{t + 1}$ and then pushed into memory pool for training.

```python
class Environment:
    def __init__(self, grid_map):
        self.grid_map = grid_map

    """
    Reset the passengers on the grid map
    """
    def reset(self):
        self.grid_map.passengers = []
        self.grid_map.add_passengers(self.grid_map.num_passengers)
```

#### $\epsilon$-greedy

$\epsilon$ is an floating point number between 0 and 1,
and this is the probability that the agent will choose random action.

Because initially the agent doesn't know how to determine the action,
this method allows the agent to learn different ways to determine action by randomly choosing it.

As the experiment progresses, the value of $\epsilon$ will keep decreasing until it reach the minimum value.

By using this method, the agent will tend to randomly choose action at beginning,
and will tend to determine action by what it learned at the end.

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
* Efficient Ridesharing Dispatch Using Multi-Agent Reinforcement Learning
* https://www.freecodecamp.org/news/dijkstras-algorithm-explained-with-a-pseudocode-example/
