# Reinforcement Learning in C++ (Implementation focused)

## Introduction

As the name suggest this blog is about showcasing and explaining how you could use Reinforcement Learning with C++ while still comparing it to its python alternative and also briefly explaining some Reinforcement Learning concepts for a better understanding of the code. There will also be exaples of how you could build training environments and how a reward function for an environment will look but the explanations about why I did it like and what alternatives would there be will be keept to a minimum.

## What should you expect to learn from this blog?

After reading this blog and the associated materials you should be able to succesfully implement some form of Deep-Learning or DQN algorithms for C++ and be able to set up pytorch for visual studio C++ projects.

## Deep-Learning with the Frozen Lake environment

### Some usefull materials: [Hugging Face Deep RL course (Deep-Learning with Frozen Lake)](https://huggingface.co/learn/deep-rl-course/unit2/hands-on) and [RL bool for beginners](http://incompleteideas.net/book/RLbook2020.pdf)


We will start by initializing the values for the Deep Learning algorithm. The environment that we will be using will be a grid of 4X4.

```cpp
    float l_epsilon = 0.9f; // this means that there is a 90% change to select a random action
    const float max_epsilon = 1.0f; // epsilon initial value
    const float min_epsilon = 0.05f; // epsilon final value (5% chance for a random action)
    const float decay_rate = 0.0005f; // the decay rate for epsilon

    const int total_episodes = 1000; // the number of episodes that the model will be trained for
    const int n_eval_episodes = 100; // this is for evaluating from 100 to 100 episodes during training
    const int max_steps = 99; // the maximum number of steps that an agent can take inside the environment
    const float lr_rate = 0.7f; // the learning rate for the Q table
    const float gamma = 0.95f;
```
Next I will create the function that we will use in order to get a random action
```cpp
    int RandomInt(int min, int max) { return min + (rand() % static_cast<int>(max - min + 1)); }
```

Now it is time to start defining the training environment. We will start by creatin the grid world and setting our position in the top left corner of the grid. The grid values signifie as followed:
- 0 is an empty space
- 1 is a hole in the ice, in this case similar to a trap
- 2 is where the agent needs to go
- 3 is where the player starts
```cpp
    int target = 15;
    int tiles[4][4] = { {3, 0, 0, 0}, {0, 1, 0, 1}, {0, 0, 0, 1}, {1, 0, 0, 2} };
```
Now we need to create the Q-Table and set the actiona and observation spaces. The observation space in general is what the agent sees while training and the action space is what action he can performe. As for the Q-Table, the Q-Table represents a grid like space where we map the value of taking any possible action in a certain state of the environment, like for example we could be in our initial position from where we can performe 4 movement: go up, go down, go left and go right. The algorithm will fill the Q-Table with values based on how good an action is in a certain state.
```cpp
    const int observation_space=16, action_space=4;

    float q_table[observation_space][action_space] = {0.0f};
```

Then we will create some functions to help us during the training.


Will start with the structure of the information that we get when we take an action. This struct will tell us in which state are we after taking the action, the action that we took, the reward that we get from the environment for taking that action and if the simulation is done.
```C++
struct step_return_values
{
public:
    int state;
    int action;
    float reward;
    bool done;
}step_values;
```

Now we will create a function for initializing the Q-Table.
```cpp
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
```
Then one function for getting the best action that we could take in a specific state based on the Q-Table.
```cpp
int q_table_max_row_value(int state) {
    float value = q_table_max_row_reward(state);
    if (value == q_table[state][0]) return 0;
    if (value == q_table[state][1]) return 1;
    if (value == q_table[state][2]) return 2;
    if (value == q_table[state][3]) return 3;
    return -1;
}

float q_table_max_row_reward(int state) {
    return std::max(std::max(q_table[state][0], q_table[state][1]), std::max(q_table[state][2], q_table[state][3]));
}
```
Now we will create the function responsible for giving us an action. This will be based on the greedy policy and epsilon greedy policy where we will decide an action based on a give epsilon and the state. Epsilon will define how often we want to take a random action. In the initial stage we want the agent to try random action until he starts to learn from them, and with each action we want to lower epsilon.
```cpp
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


Now lets see how the update function and the reward system for this specific environment will look.
The actioned are as followed:
- 0 move right
- 1 move up
- 2 move left
- 3 move down
As for the reward, the agent will get a positive reward if it finishes the map and if it hits a trap the simulation will be ended. The reset function will reset the environment to its initial state.
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

Now lets jump to putting together the Deep-Learning algorithm.

```C++
void main()
{
    // Here we will create the values that we will need
    int episode = 0;
    int t = max_steps;
    bool done = false;
    int state = 0;
    int action = 0;

    // Here is where the training begins
    for(int episode = 0; episode <= total_episodes; episode++){
      for(int t = 0; t<max_steps && !done; t++)
      {
        // We select an action
        action = epsilon_greedy_policy(state);

        // Then we take that action in our current state
        step_values = env_step(state,action);

        // And finally we evaluate the action that we took in the previouse state
        q_table[state][action] = q_table[state][action] + lr_rate * (step_values.reward + gamma * q_table_max_row_reward(step_values.state) -  q_table[state][action]);

        done = step_values.done;
        // Setting the environment to its current state after taking the action is an easy to miss step so be carefull
        state = step_values.state;
        t++;
      }
        // Here we calculate the epsilon based on the current episode
      l_epsilon = min_epsilon + (max_epsilon - min_epsilon) * pow(M_E, static_cast<float>(episode) * -decay_rate);

        //Then we reset the environment
      state = reset_env();
      done = false;
      t = 0;
}
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

The result should be something like this:

![video](/Images/Demo.gif)

## DQN with Moon Lander and Racing Track

Now that we are done with our first RL algorithm you might have already seen the drawbacks of Deep-Learning such as the fact that you would need to create a Q-table based on all the states that an environemnt could happen to be in and the actions that could be taken in that mentioned state. This could get really hard or close to impossible to represent with an environment that is in a continuous state such as a racing game for example. This is where DQN comes into place by using a network to approximate the value of taking an action a in a state s.

For the start lets get starting with the DQN algorithm first, but before that you will need to set up pytorch (the network library) for visual studion. This should show you [how to set it up](https://khushi-411.github.io/setting-pytorch-api-c++/), [this is for downloading pytorch](https://pytorch.org/get-started/locally/) and this is the [python tutorial](https://anvilproject.org/guides/content/creating-links) that I based my arhitecture on. An always usefull link is the [pytorch C++ documentation](https://pytorch.org/cppdocs/frontend.html).

Now that we have everything that we need lest jump into it

We'll start with the headers that we will need first
```C++
#pragma once
#include <torch/nn.h>
#include <torch/nn/functional.h>
#include <torch/nn/module.h>
#include <torch/optim.h>
#include <torch/torch.h>
#include <torch/serialize/output-archive.h>

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <random>
```
Then we will create the structs that will we will use for storing the state and the step return information when working with the DQN algorithm
```cpp
struct Float_State
{
    std::vector<float> state;
};

struct Float_Step_Return
{
    Float_State state;
    Float_State next_state;
    float action;
    float reward;
    float terminated;
    float truncated;
};

struct Full_Float_Step_Return
{
    std::vector<float> data;
};

struct Tensor_step_return
{
    torch::Tensor states;
    torch::Tensor actions;
    torch::Tensor next_states;
    torch::Tensor rewards;
    torch::Tensor dones;
};
```
And some values that we will be using later
```cpp
unsigned seed;
const int BUFFER_SIZE = int(2e5);
const int BATCH_SIZE = 64;
const float GAMMA = 0.99f;
float TAU = 1e-3;
float LR = 5e-4;
int UPDATE_EVERY = 32;

```

Now that we have the structs that we will need ready, lets start with the Network class first
```cpp
class QNetworkImpl : public torch::nn::Module
{
public:
    QNetworkImpl(int state_size, int action_size, int seed){
    //fc1 tp fc2 are hidden layers for the network
    torch::manual_seed(seed);
    fc1 = register_module("fc1", torch::nn::Linear(state_size, 128));
    fc2 = register_module("fc2", torch::nn::Linear(128, 128));
    fc3 = register_module("fc3", torch::nn::Linear(128, action_size));
};

    QNetworkImpl(int state_size, int action_size){ QNetwork(state_size, action_size, 0); };

    QNetworkImpl(){};

    // This function will return us the values for each action that the network computem with the given state
    torch::Tensor forward(torch::Tensor x){
    x = fc1(x);
    x = torch::relu(x);
    x = fc2(x);
    x = torch::relu(x);
    x = fc3(x);
    return x;
};

// This resets the network layers to their initial values
    void resetNetwork(){
    for (auto& layer : this->children())
    {
        layer.reset();
    }
};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

TORCH_MODULE(QNetwork);
```

Then we will create the memory buffer class which will be responsible for sampling previouse random states of the environment.

```C++
class ReplayBuffer
{
public:
   ReplayBuffer(int state_size, int action_size, int buffer_size, int batch_size, int seed)
{
    this->state_size = state_size;
    this->action_size = action_size;
    this->buffer_size = buffer_size;
    this->batch_size = batch_size;
    this->seed = seed;
};

ReplayBuffer(int action_size, int buffer_size, int batch_size)
{
    this->buffer_size = buffer_size;
    this->batch_size = batch_size;
    this->seed = seed;
};

ReplayBuffer(){};

void add(Full_Float_Step_Return experience)
{
    experiences.push_back(experience);
    if (experiences.size() > buffer_size) experiences.erase(experiences.begin());
};

void addBulk(std::vector<Full_Float_Step_Return>& experiences)
{
    this->experiences.insert(this->experiences.end(), std::make_move_iterator(experiences.begin()),
                             std::make_move_iterator(experiences.end()));
    if (this->experiences.size() > buffer_size)
    {
        this->experiences.erase(this->experiences.begin(),
                                this->experiences.begin() + (this->experiences.size() - buffer_size));
    }
};

// This is the most important function from this class, which will sample us batch_size random experiences in a random order, and convert them to tensors (the general container used in pytorch) so that the network could read them and use them for learning 
Tensor_step_return sample()
{
    Tensor_step_return tensor;
    std::vector<Full_Float_Step_Return> batch;
    std::sample(experiences.begin(), experiences.end(), std::back_inserter(batch), batch_size,
                std::mt19937{std::random_device{}()});
    std::shuffle(batch.begin(), batch.end(), std::mt19937{std::random_device{}()});

    rewards.clear();
    actions.clear();
    dones.clear();

    torch::Tensor ns_tensor, s_tensor;

    int i = 0;
    for (auto& experience : batch)
    {
        i++;
        s_tensor = torch::from_blob((float*)(experience.data.data()), state_size);
        if (i > 1)
            tensor.states = torch::cat({tensor.states, s_tensor.unsqueeze(0)}, 0);
        else
            tensor.states = torch::cat({s_tensor.unsqueeze(0)}, 0);

        s_tensor = torch::from_blob((float*)(experience.data.data() + state_size), state_size);
        if (i > 1)
            tensor.next_states = torch::cat({tensor.next_states, s_tensor.unsqueeze(0)}, 0);
        else
            tensor.next_states = torch::cat({s_tensor.unsqueeze(0)}, 0);

        rewards.push_back(experience.data[state_size * 2]);
        actions.push_back(experience.data[state_size * 2 + 1]);
        dones.push_back(experience.data[state_size * 2 + 2]);
        if (i == BATCH_SIZE) break;
    }

    tensor.actions = torch::from_blob((float*)(actions.data()), actions.size()).unsqueeze(1);
    tensor.rewards = torch::from_blob((float*)(rewards.data()), rewards.size()).unsqueeze(1);
    tensor.dones = torch::from_blob((float*)(dones.data()), dones.size()).unsqueeze(1);
    return tensor;
};

    int state_size;
    int action_size;
    int buffer_size;
    int batch_size;

    std::vector<Full_Float_Step_Return> experiences;
    int seed;

    std::vector<float> actions;
    std::vector<float> rewards;
    std::vector<float> dones;
};
```
And lastly the DQN algorithm class
```C++
class DQN
{
public:
    DQN(int state_size, int action_size, int seed)
{
    this->state_size = state_size;
    this->action_size = action_size;
    this->seed = seed;

    q_network = QNetwork(state_size, action_size, seed);
    fixed_network = QNetwork(state_size, action_size, seed);
    auto adamOptions = torch::optim::AdamOptions(0.0001);
    optimizer = new torch::optim::Adam(q_network->parameters(), adamOptions);
    buffer = ReplayBuffer(state_size, action_size, BUFFER_SIZE, BATCH_SIZE, seed);
};

DQN(int state_size, int action_size) { DQN(state_size, action_size, 0); };

DQN(){};

//This will update the network every type we have enough data in the buffer
void step()
{
    if (timestep >= UPDATE_EVERY)
    {
        if (buffer.experiences.size() > BATCH_SIZE)
        {
            Tensor_step_return sampled_experiences = buffer.sample();
            // printf("%i\n",buffer.experiences.size());
            learn(sampled_experiences);
        }
        timestep = timestep % UPDATE_EVERY;
    }
};

void addToExperienceBuffer(Full_Float_Step_Return value)
{
    buffer.add(value);  //(state, action, reward, next_state, done);
    timestep++;
};

void addToExperienceBufferInBulk(std::vector<Full_Float_Step_Return>& values)
{
    timestep += values.size();
    buffer.addBulk(values);
};

// This will give us an action random or from the network based on the value of epsilon and a random number
int act(Float_State state, float epsilon)
{
    torch::NoGradGuard no_grad;

    int action;
    float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    if (r > epsilon)
    {
        torch::Tensor t_state = torch::from_blob((float*)(state.state.data()), state_size).unsqueeze(0);
        torch::Tensor action_values;

        action_values = q_network->forward(t_state);
        action = static_cast<int>(torch::argmax(action_values).item().toInt() % action_size);
        currentStep++;
        // std::cout << t_state << "\n" << action_values << "\n" << action << "\n\n";
        currentStep -= whenToPrint;
    }
    else
    {
        action = static_cast<int>(rand() % action_size);
    }
    // std::cout << "\n" << action << "\n";
    return action;
};

// This will update the network layers values based on the experiences that we sampled and a fixed network that contains the previous state of the network
void learn(Tensor_step_return experiences)
{
    // this->q_network->train();

    torch::Tensor action_values;
    torch::Tensor max_action_values;

    {
        torch::NoGradGuard no_grad;

        action_values = fixed_network->forward(experiences.next_states).detach();
        auto [ttt, stuff] = action_values.max(1);
        max_action_values = ttt.unsqueeze(1);
    }

    torch::Tensor Q_target = experiences.rewards + (GAMMA * max_action_values * (1 - experiences.dones));
    torch::Tensor Q_expected = q_network->forward(experiences.states).gather(1, experiences.actions.to(torch::kLong));
    torch::Tensor loss = torch::nn::functional::mse_loss(Q_expected, Q_target);
    // std::cout << Q_target << "\n" << Q_expected << "\n" << loss << "\n";
    optimizer->zero_grad();

    loss.backward();
    optimizer->step();

    update_fixed_network(q_network, fixed_network);

    // this->q_network->eval();
};

// This updates the fixed network with the last values of the current network
void update_fixed_network()
{
    torch::NoGradGuard no_grad;

    for (int i = 0; i < q_network->parameters().size(); i++)
    {
        fixed_network->parameters()[i].data().copy_(TAU * q_network->parameters()[i].data() +
                                                    (1.0f - TAU) * fixed_network->parameters()[i].data());
    }
};

// This creates a checkpoint of the training that can be loaded and used later
void checkpoint(std::string filepath)
{
    torch::save(q_network, (filepath + "_network.pt").c_str());
    torch::save(*optimizer, (filepath + "_optimizer.pt").c_str());
};

// This loads a previously saved checkpoint of the training
void loadCheckpoint(std::string filepath)
{
    torch::load(q_network, (filepath + "_network.pt").c_str());
    torch::load(*optimizer, (filepath + "_optimizer.pt").c_str());
};

// This resets the training
void resetLearning()
{
    q_network->resetNetwork();
    fixed_network->resetNetwork();
    delete optimizer;
    auto adamOptions = torch::optim::AdamOptions(0.0001);
    optimizer = new torch::optim::Adam(q_network->parameters(), adamOptions);
};

    int state_size, action_size, seed;

    QNetwork q_network, fixed_network;
    torch::optim::Adam* optimizer;

    ReplayBuffer buffer;
    int timestep = 0;

    int whenToPrint = 1000;
    int currentStep = 0;
};
```

Now lets see how we will used everything above.

First we declare some values similar to the Deep-Learning example

```C++
const int max_episodes = 2000;
const int max_steps = 2000;
const int print_every = 100;

const float eps_start = 1.0f;
const float eps_decay = 100000;
const float eps_min = 0.01f;
```

Then we create the environment and the DQN class
```cpp
 agent = new DQN(8, 4, 0);// state size = 8 and action size = 4 for Lunar Lander
 env = new Racing_Track(projectionDimensions);
```
Some variables that we will use later
```cpp
Action action = Action::Nothing;

float score = 0;
int t = max_steps;

float eps = eps_start;

State state;

Step_return step_return;
```
And now finnaly the training
```cpp
   for(int episode = 0; episode <= max_episodes; episode++)
   {
       while (t > 0 && !done)
       {
        eps = eps_min + (eps_start - eps_min) * exp(-1. * stepsDone / eps_decay);
        action = static_cast<Action>(agent->act(envs[i].env->StateToFloat_State(envs[i].env->squizzForNetwork(envs[i].env->currentState)), eps));
        step_return = envs[i].env->step(/*dt*/ 0.005f, action);

        agent->addToExperienceBuffer(envs[i].env->StepReturnToFullFLoatStepReturn(step_return));//adds the experience to the buffer
        agent->step(); //updates the network
        done = step_return.terminated;
        t--;
       }
           if (episode % print_every == 0)
           {
               path = path + std::to_string(episode) + ".pth";
               agent->checkpoint(path);
           }
           env->reset();
           score = 0.0f;
           done = false;
           t = max_steps;
       }
   }
```
The results would be something like this.
![video](/Images/BeforeOptimization.gif)

Now if we want to speed up the training, we could train with multiple sets of data at once, like this.

```C++
const int max_episodes = 2000;
const int max_steps = 2000;
const int print_every = 100;

const float eps_start = 1.0f;
const float eps_decay = 100000;
const float eps_min = 0.01f;

const int nEnv = 32;

struct Environment{
 Lunar_Lander* env;
 bool done = false;
int steps = 0;
};

 agent = new DQN(8, 4, 0);// state size = 8 and action size = 4 for Lunar Lander
std::vector<Environment> envs;

for(int i=0;i<nEnv;i++){
envs.push_back(Environment{});//projection dimension is half of the game screen width and half of the screen height
envs[i].env = new Lunar_Lander(projectionDimensions);
}
Action action = Action::Nothing;

float score = 0;
int t = max_steps;

float eps = eps_start;

State state;
bool done;

Step_return step_return;

   for(int episode = 0; episode <= max_episodes; episode++)
   {
    done = true;
    for (int i = 0; i < nEnv; i++)
    {
        done = done && envs[i].done;
    }
    while (!done)
    {
        for (int i = 0; i < nEnv; i++)
        {
            if (!envs[i].done)
            {
                eps = eps_min + (eps_start - eps_min) * exp(-1. * stepsDone / eps_decay);
                action = static_cast<Action>(agent->act(
                envs[i].env->StateToFloat_State(envs[i].env->squizzForNetwork(envs[i].env->currentState)), eps));
                step_return = envs[i].env->step(0.005f, action);
                steps.push_back(envs[i].env->StepReturnToFullFLoatStepReturn(step_return));
                envs[i].steps++;

                if (step_return.terminated || envs[i].steps >= max_steps)
                {
                    envs[i].done = true;
                }
            }
        }
        agent->addToExperienceBufferInBulk(steps);
        agent->step();
    }
    for (int i = 0; i < nEnv; i++)
    {
        envs[i].env->reset();
        envs[i].done = false;
        envs[i].steps = 0;
    }

    if (episode % print_every == 0)
    {
        std::string path;
        cout << mean_score << "\n";
        path = "Checkpoints/" + std::to_string(episode);
        agent->checkpoint(path);
    }
   }
```
The results would be something like this.
![video](/Images/TrainingOptimization.gif)

After training for some time you should get a model that behaves like this.
The results would be something like this.
![video](/Images/TrainedModel.gif)
![video](/Images/TrainedModel2.gif)

And this is an example of how you could load a checkpoint for the DQN agent
```C++
std::string _path = "200";
_path = "Checkpoints/" + _path;
agent->loadCheckpoint(_path);
```
And this is how to use the agent after training

```C++
agent->q_network->eval();
Float_State state = env->StateToFloat_State(env->squizzForNetwork(env->currentState));
torch::Tensor s_tensor = torch::from_blob((float*)(state.state.data()), agent->state_size).unsqueeze(0);

torch::Tensor action_values = agent->q_network->forward(s_tensor);
action = static_cast<Action>(torch::argmax(action_values).item().toInt() % agent->action_size);
dt = 0.005f;
env->step(0.005f, action);
```
To conclude everything, even though some parts might seem confusing the purpose of this blog was to show you how the implementation would look. If you want to have a better understanding over what we went through just take you time, read through the materials recomended and come back after you have a better understanding over the theory behing DQN and Deep-Learning, or in general over RL algorithms.


