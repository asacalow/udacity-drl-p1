# README

Here be my solution to the first Navigation Project in Udacity's Deep Reinforcement Learning course.

## Project Details

The environment to solve is the Banana Collector Unity environment for the p1_navigation project, described in full in the environment's README [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation). For convenience, the relevant infos on the state and action spaces are reproduced below:

> A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

> The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

> 0 - move forward.
> 1 - move backward.
> 2 - turn left.
> 3 - turn right.

> The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

## Getting Started

The only additional dependency for this solution, beyond the standard project dependencies (as described in the [main Udacity course repo](https://github.com/udacity/deep-reinforcement-learning)) is Pytorch's Ignite framework, which can be installed via either pip or conda as described [here](https://pytorch.org/ignite/).

## Training the agent

Training the agent is achieved by running the Reinforce-Unity-PER-Dueling notebook in this repo. There are a couple more implementations – one using plain Double DQN, and another frankly terrible one which applies both Double DQN and PER. Ignore these and go straight for Reinforce-Unity-PER-Dueling – this is where you'll find the (reasonably) good stuff.