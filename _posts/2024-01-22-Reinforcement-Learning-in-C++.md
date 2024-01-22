# Reinforcement Learning in C++ (Implementation focused)

## Introduction

As the name suggest this blog is about showcasing and explaining how you could use Reinforcement Learning with C++ while still comparing it to its python alternative and also briefly explaining some Reinforcement Learning concepts for a better understanding of the code. There will also be exaples of how you could build training environments and how a reward function for an environment will look but the explanations about why I did it like and what alternatives would there be will be keept to a minimum.

## What should you expect to learn from this blog?

After reading this blog and the associated materials you should be able to succesfully implement some form of Deep-Learning or DQN algorithms for C++, be able to set up pytorch for visual studio C++ projects and also be able to create your own environment or use some built by others.

## Deep-Learning with the Frozen Lake environment

### Some usefull materials: [Hugging Face Deep RL course (Deep-Learning with Frozen Lake)](https://huggingface.co/learn/deep-rl-course/unit2/hands-on) and [RL bool for beginners](http://incompleteideas.net/book/RLbook2020.pdf)


We will start by initializing the values for the Deep Learning algorithm. The environment that we will be using will be a grid of 4X4.
```C++
float l_epsilon = 0.9f; // this means that there is a 90% change to select a random action
const float max_epsilon = 1.0f; // epsilon initial value
const float min_epsilon = 0.05f; // epsilon final value (5% chance for a random action)
const float decay_rate = 0.0005f; // the decay rate for epsilon

const int total_episodes = 1000; // the number of episodes that the model will be trained for
const int n_eval_episodes = 100; // this is for evaluating from 100 to 100 episodes during training
const int max_steps = 99; // the maximum number of steps that an agent can take inside the environment
const float lr_rate = 0.7f; // the learning rate for the Q table
const float gamma = 0.95f;

int RandomInt(int min, int max) { return min + (rand() % static_cast<int>(max - min + 1)); }

int target = 0;
int tiles[4][4] = {{3, 0, 0, 0}, {0, 1, 0, 1}, {0, 0, 0, 1}, {1, 0, 0, 2}};
const int observation_space=16, action_space=4;

float q_table[observation_space][action_space] = {0.0f};
```

Then we will create some functions to help us during the training.

```C++
struct step_return_values
{
public:
    int state;
    int action;
    float reward;
    bool done;
}step_values;

void init_q_table()
{
    for (int i = 0; i < observation_space; i++)
    {
        for (int j = 0; j < j < action_space; j++)
        {
            q_table[i][j] = 0.0f;
        }
    }
}

int q_table_max_row_value(int state) {
    float value = std::max(std::max(q_table[state][0], q_table[state][1]), std::max(q_table[state][2], q_table[state][3]));
    if (value == q_table[state][0]) return 0;
    if (value == q_table[state][1]) return 1;
    if (value == q_table[state][2]) return 2;
    if (value == q_table[state][3]) return 3;
    return -1;
}

float q_table_max_row_reward(int state) {
    return std::max(std::max(q_table[state][0], q_table[state][1]), std::max(q_table[state][2], q_table[state][3]));
}

int greedy_policy(int state) {

    return q_table_max_row_value(state);
}

int epsilon_greedy_policy(int state)
{
    int action;
    float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    if (r > l_epsilon)
    {
        action = greedy_policy(state);
    }
    else
    {
        action = rand() % 4;
    }
    return action;
}
```


This is implementation of Deep-Learning works by reseting the environment to its initial state at the start of each episode.
During the episode, in each step you will get an action based on the current state of the environmen from the greedy policy. Then you update the environment based on the action that you get and lastly you update the Q-table based on the previouse state, current state, the action that you took and the reward that you get from the environmen for that action.

```C++
void main()
{
    // Move player
    int episode = 0;
    int t = max_steps;
    bool done = false;
    int state = 0;
    int action = 0; 
    for(int episode = 0; episode < total_episodes; episode++){
      for(int t = 0; t<max_steps && !done; t++)
      {
        action = epsilon_greedy_policy(state);
        step_values = env_step(state,action);
        q_table[state][action] = q_table[state][action] + lr_rate * (step_values.reward + gamma * q_table_max_row_reward(step_values.state) -  q_table[state][action]);
        done = step_values.done;
        state = step_values.state;
        t++;
      }
      l_epsilon = min_epsilon + (max_epsilon - min_epsilon) * pow(M_E, static_cast<float>(episode) * -decay_rate);
      state = reset_env();
      episode++;
      done = false;
      t = 0;
}
return 0;
}
```

Lastly this is how the reward function and the environment will look.

```C++
step_return_values env_step(int state,int action) {
    int l_state=0;
    switch (action)
    {
        case 0:
        {
            if (state % 4 + 1 > 3)
                l_state = state;
            else
                l_state = state + 1;
            break;
        }
        case 1:
        {
            if (state / 4 - 1 < 0)
                l_state = state;
            else
                l_state = state - 4;
            break;
        }
        case 2:
        {
            if (state % 4 - 1 < 0)
                l_state = state;
            else
                l_state = state - 1;
            break;
        }
        case 3:
        {
            if (state / 4 + 1 > 3)
                l_state = state;
            else
                l_state = state + 4;
            break;
        }
        default:
        {
            break;
        }
    }
    bool done = (l_state == target);
    float reward = 0.0f;
    if (done)
    {
        reward = 1.0f;
    }
    if (tiles[l_state / 4][l_state % 4] == 1)
    {
        done = true;
        reward = 0.0f;
    }
    return step_return_values{l_state,0,reward,done};
}

int reset_env() {
    return 0;
}
```

As for using the model, you could do something like this after training.
```C++
l_epsilon = 0.0f;
for(int t = 0; t<max_steps && !done; t++)
      {
        action = epsilon_greedy_policy(state);
        step_values = env_step(state,action);
        done = step_values.done;
        state = step_values.state;
        t++;
      }     
```

## DQN with Moon Lander and Racing Track

Now that we are done with our first RL algorithm you might have already seen the drawbacks of Deep-Learning such as the fact that you would need to create a Q-table based on all the states that an environemnt could happen to be in and the actions that could be taken in that mentioned state. This could get really hard or close to impossible to represent with an environment that is in a continuous state such as a racing game for example. This is where DQN comes into place by using a network to approximate the value of taking an action a in a state s.

For the start lets get starting with the DQN algorithm first, but before that you will need to set up pytorch (the network library) for visual studion. This should show you [how to set it up](https://khushi-411.github.io/setting-pytorch-api-c++/) and [this is for downloading pytorch](https://pytorch.org/get-started/locally/). An always usefull link is the [pytorch C++ documentation](https://pytorch.org/cppdocs/frontend.html).

Now that we have everything that we need lest jump into it

We'll start with the network first.
```C++
#pragma once
#include <torch/nn.h>
#include <torch/nn/functional.h>
#include <torch/nn/module.h>
#include <torch/optim.h>
#include <torch/torch.h>

struct Tensor_step_return
{
    torch::Tensor states;
    torch::Tensor actions;
    torch::Tensor next_states;
    torch::Tensor rewards;
    torch::Tensor dones;
};

class QNetworkImpl : public torch::nn::Module
{
public:
    QNetworkImpl(int state_size, int action_size, int seed);
    QNetworkImpl(int state_size, int action_size);
    QNetworkImpl(){};
    torch::Tensor forward(torch::Tensor x);
    void resetNetwork();

    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

TORCH_MODULE(QNetwork);
```
```C++
QNetworkImpl::QNetworkImpl(int state_size, int action_size, int seed)
{
    torch::manual_seed(seed);
    fc1 = register_module("fc1", torch::nn::Linear(state_size, 128));
    fc2 = register_module("fc2", torch::nn::Linear(128, 128));
    fc3 = register_module("fc3", torch::nn::Linear(128, action_size));
}

QNetworkImpl::QNetworkImpl(int state_size, int action_size) { QNetwork(state_size, action_size, 0); }

torch::Tensor QNetworkImpl::forward(torch::Tensor x)
{
    x = fc1(x);
    x = torch::relu(x);
    x = fc2(x);
    x = torch::relu(x);
    x = fc3(x);
    return x;
}

void QNetworkImpl::resetNetwork()
{
    for (auto& layer : this->children())
    {
        layer.reset();
    }
}
```
