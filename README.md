# ORA_final_project
## Content
* Motivation and background
* Methodology
## Methodology
**Reinforcement Learning**

RL is known as a semi-supervised learning model in machine learning which enables an agent to learn in an interactive environment by trial and error using feedback from its own actions and experiences.

Unlike supervised learning where the feedback provided to the agent is correct set of actions for performing a task, RL uses rewards and punishments as signals for positive and negative behavior. Thus, its goal is to find a suitable action model that would maximize the total cumulative reward of the agent.

![Image text](https://github.com/Hamu111268/ORA_final_project/blob/main/img_storage/picture1.png)

**Q-Learning**

Q-learning is a model-free reinforcement learning algorithm to learn the value of an action in a particular state.

In Q-learning, the experience learned by the agent is stored in the Q table, and the value in the table expresses the long-term reward value of taking specific action in a specific state. According to the Q table, the Q learning algorithm can tell the Q agent which action to choose in a specific situation to get the largest expected reward.

![Image text](https://github.com/Hamu111268/ORA_final_project/blob/main/img_storage/picture2.png)

In a certain iteration, the agent of the Q-learning algorithm is in the state s_t at the time t and chooses an action a_t according to the strategy. It receives the reward r_t from the environment and enters the new state s_(t+1), and then the Q table is updated according to the equation.

**Deep Q-Learning**

When the problem scale is larger, Q-table will be inefficient. Therefore, Deep Q-learning had been developed to use a neural network to approximate the Q-value function. The state is given as the input and the Q-value of all possible actions is generated as the output.

![Image text](https://github.com/Hamu111268/ORA_final_project/blob/main/img_storage/picture3.png)
