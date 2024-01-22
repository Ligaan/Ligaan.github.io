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

For the start lets get starting with the DQN algorithm first, but before that you will need to set up pytorch (the network library) for visual studion. This should show you [how to set it up](https://khushi-411.github.io/setting-pytorch-api-c++/), [this is for downloading pytorch](https://pytorch.org/get-started/locally/) and this is the [python tutorial](https://anvilproject.org/guides/content/creating-links) that I based my arhitecture on. An always usefull link is the [pytorch C++ documentation](https://pytorch.org/cppdocs/frontend.html).

Now that we have everything that we need lest jump into it

We'll start with the network first.
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
Then we will create the memory buffer class which will be responsible for sampling previouse random states of the environment.

```C++
class ReplayBuffer
{
public:
    ReplayBuffer(int state_size, int action_size, int buffer_size, int batch_size, int seed);
    ReplayBuffer(int action_size, int buffer_size, int batch_size);
    ReplayBuffer(){};

    void add(Full_Float_Step_Return experience);  //(State state, Action action, float reward, State next_state, bool done);
    void addBulk(std::vector<Full_Float_Step_Return>& experiences);
    Tensor_step_return sample();

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
```C++
ReplayBuffer::ReplayBuffer(int state_size, int action_size, int buffer_size, int batch_size, int seed)
{
    this->state_size = state_size;
    this->action_size = action_size;
    this->buffer_size = buffer_size;
    this->batch_size = batch_size;
    this->seed = seed;
}

ReplayBuffer::ReplayBuffer(int action_size, int buffer_size, int batch_size)
{
    this->buffer_size = buffer_size;
    this->batch_size = batch_size;
    this->seed = seed;
}

void ReplayBuffer::add(Full_Float_Step_Return experience)
{
    experiences.push_back(experience);
    if (experiences.size() > buffer_size) experiences.erase(experiences.begin());
}

void ReplayBuffer::addBulk(std::vector<Full_Float_Step_Return>& experiences)
{
    this->experiences.insert(this->experiences.end(), std::make_move_iterator(experiences.begin()),
                             std::make_move_iterator(experiences.end()));
    if (this->experiences.size() > buffer_size)
    {
        this->experiences.erase(this->experiences.begin(),
                                this->experiences.begin() + (this->experiences.size() - buffer_size));
    }
}

Tensor_step_return ReplayBuffer::sample()
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
}
```
And lastly the DQN algorithm class
```C++
class DQN
{
public:
    DQN(int state_size, int action_size, int seed);
    DQN(int state_size, int action_size);
    DQN(){};
    void step();  //(State state, Action action, float reward, State next_state, bool done);
    void addToExperienceBuffer(Full_Float_Step_Return value);
    void addToExperienceBufferInBulk(std::vector<Full_Float_Step_Return>& values);
    int act(Float_State state, float epsilon);
    void learn(Tensor_step_return experiences);
    void update_fixed_network(QNetwork& local_model, QNetwork& target_model);
    void checkpoint(std::string filepath);
    void loadCheckpoint(std::string filepath);
    void resetLearning();

    int state_size, action_size, seed;

    QNetwork q_network, fixed_network;
    torch::optim::Adam* optimizer;

    ReplayBuffer buffer;
    int timestep = 0;

    int whenToPrint = 1000;
    int currentStep = 0;
};
```
```C++
unsigned seed;
const int BUFFER_SIZE = int(2e5);  // 1e5
const int BATCH_SIZE = /*128*/ 64;
const float GAMMA = 0.99f;
float TAU = 1e-3;
float LR = 5e-4;
int UPDATE_EVERY = 32; /*16;*/

DQN::DQN(int state_size, int action_size, int seed)
{
    this->state_size = state_size;
    this->action_size = action_size;
    this->seed = seed;

    q_network = QNetwork(state_size, action_size, seed);
    fixed_network = QNetwork(state_size, action_size, seed);
    auto adamOptions = torch::optim::AdamOptions(0.0001);
    optimizer = new torch::optim::Adam(q_network->parameters(), adamOptions);
    buffer = ReplayBuffer(state_size, action_size, BUFFER_SIZE, BATCH_SIZE, seed);
}

DQN::DQN(int state_size, int action_size) { DQN(state_size, action_size, 0); }

void DQN::step()
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
}

void DQN::addToExperienceBuffer(Full_Float_Step_Return value)
{
    buffer.add(value);  //(state, action, reward, next_state, done);
    timestep++;
}

void DQN::addToExperienceBufferInBulk(std::vector<Full_Float_Step_Return>& values)
{
    timestep += values.size();
    buffer.addBulk(values);
}

int DQN::act(Float_State state, float epsilon)
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
}

void DQN::learn(Tensor_step_return experiences)
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
}

void DQN::update_fixed_network(QNetwork& local_model, QNetwork& target_model)
{
    torch::NoGradGuard no_grad;

    for (int i = 0; i < q_network->parameters().size(); i++)
    {
        fixed_network->parameters()[i].data().copy_(TAU * q_network->parameters()[i].data() +
                                                    (1.0f - TAU) * fixed_network->parameters()[i].data());
    }
}

void DQN::checkpoint(std::string filepath)
{
    torch::save(q_network, (filepath + "_network.pt").c_str());
    torch::save(*optimizer, (filepath + "_optimizer.pt").c_str());
}

void DQN::loadCheckpoint(std::string filepath)
{
    torch::load(q_network, (filepath + "_network.pt").c_str());
    torch::load(*optimizer, (filepath + "_optimizer.pt").c_str());
}

void DQN::resetLearning()
{
    q_network->resetNetwork();
    fixed_network->resetNetwork();
    delete optimizer;
    auto adamOptions = torch::optim::AdamOptions(0.0001);
    optimizer = new torch::optim::Adam(q_network->parameters(), adamOptions);
}
```

Now lets see how we will used everything above

```C++
const int max_episodes = 2000;
const int max_steps = 2000;
const int print_every = 100;

const float eps_start = 1.0f;
const float eps_decay = 100000;
const float eps_min = 0.01f;

 agent = new DQN(8, 4, 0);// state size = 8 and action size = 4 for Lunar Lander
 env = new Lunar_Lander(projectionDimensions);

Action action = Action::Nothing;

float score = 0;
int t = max_steps;

float eps = eps_start;

State state;

Step_return step_return;

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
Now updating the network based on the previouse experiences that we collected, so the next step would be to use multiple environments at once for training, maybe something like this.

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
Now that we are done with the DQN its time to discuse a bit about environments.

When training agent one in RL one of the most important if not the most important part of the training is the reward system. A good reward system could stimulate the agent to learn better and faster while a bad reward system could make the agent fail to learn something usefull. This shows some of the [gym environments](https://github.com/openai/gym/tree/master/gym/envs) whic might be a good space to look for examples.

Now the first example that I created is Lunar Lander

```C++
struct State
{
    float x;
    float y;
    float vx;
    float vy;
    float angle;
    float angularVelocity;
    bool leg_1_land;
    bool leg_2_land;
};

enum Action
{
    Nothing,
    Left,
    Main,
    Right
};

struct Step_return
{
    State state;
    Action action;
    State next_state;
    float reward;
    bool terminated;
    bool truncated;
};

struct Float_State;
struct Float_Step_Return;

//Float_State StateToFloat_State(State state);
//Float_Step_Return StepReturnToFloatStepReturn(Step_return step_return);

class Lunar_Lander
{
public:
    Lunar_Lander(glm::vec2 windowSize, glm::vec2 pos, float platformHeight);
    Lunar_Lander(glm::vec2 windowSize, glm::vec2 pos);
    Lunar_Lander(glm::vec2 windowSize);

    Float_State StateToFloat_State(State state);
    Float_Step_Return StepReturnToFloatStepReturn(Step_return step_return);

    void reset();
    void update(float dt, Action a);
    Step_return step(Action action);
    bool landed();
    bool landOutside();
    bool inBoundsLand(float range);
    bool inBounds(float range);
    bool crashed();
    float getShaping(float x, float y, float vx, float vy, float angle, bool leg1, bool leg2);
    State squizzForNetwork(State state);

    void setWindowSize(glm::vec2 size);

    State getState();

    glm::vec3 size = glm::vec3(10.0f, 5.0f, 3.0f);
    const float sideThrust = 2.0f;
    const float rotationForce = 0.4f;
    const float mainThrust = 16.0f;
    const float gravity = -9.8f;
    const float angularFriction = 0.98f;
    const float velocityFriction = 0.99f;
    float land_1 = -50.0f, land_2 = 0.0f;
    float x_shapingValue = 0.0f, y_shapingValue = 0.0f, x_scale = 1.0f, y_scale = 1.0f;
    float prev_shaping = 100000.0f;
    float last_reward = 0.0f;
    State currentState;
    glm::vec2 initPos;
    float platformHeight;
    const float skyLimit = 150.0f;

    glm::vec2 windowSize = glm::vec2(0.0f, 0.0f);
};
```
```C++
Lunar_Lander::Lunar_Lander(glm::vec2 windowSize, glm::vec2 pos, float platformHeight)
    : windowSize(windowSize), initPos(pos), platformHeight(platformHeight)
{
    x_shapingValue = ((land_1 + land_2) * 0.5f) / windowSize.x;
    y_shapingValue = platformHeight / windowSize.y;

    reset();
}

Lunar_Lander::Lunar_Lander(glm::vec2 windowSize, glm::vec2 pos) : windowSize(windowSize), initPos(pos), platformHeight(-100.0f)
{
    x_shapingValue = ((land_1 + land_2) * 0.5f) / windowSize.x;
    y_shapingValue = platformHeight / windowSize.y;

    reset();
}

Lunar_Lander::Lunar_Lander(glm::vec2 windowSize)
    : windowSize(windowSize), initPos(glm::vec2(0.0f, 50.0f)), platformHeight(-100.0f)
{
    x_shapingValue = ((land_1 + land_2) * 0.5f) / windowSize.x;
    y_shapingValue = platformHeight / windowSize.y;

    reset();
}

void Lunar_Lander::reset()
{
    currentState.x = initPos.x;
    currentState.y = initPos.y;
    currentState.angle = 0.0f;
    currentState.angularVelocity = 0.0f;
    currentState.leg_1_land = false;
    currentState.leg_2_land = false;
    currentState.vx = 0.0f;
    currentState.vy = 0.0f;

    prev_shaping = 100000.0f;
}

void Lunar_Lander::update(float dt, Action a)
{
    glm::vec2 l_pos = glm::vec2(currentState.vx, currentState.vy);
    glm::vec2 l_velocity = glm::vec2(0.0f, 0.0f);
    float l_angle = currentState.angularVelocity;
    float l_angularVel = 0.0f;
    l_velocity += glm::vec2(0.0f, gravity);
    switch (a)
    {
        case Action::Nothing:
        {
            break;
        }
        case Action::Left:
        {
            glm::vec2 left(-1.0f, 0.0f);
            left = glm::rotate(left, currentState.angle);
            left *= sideThrust;
            l_velocity += left;
            l_angularVel -= rotationForce;
            break;
        }
        case Action::Main:
        {
            glm::vec2 main(0.0f, 1.0f);
            main = glm::rotate(main, currentState.angle);
            main *= mainThrust;
            l_velocity += main;
            break;
        }
        case Action::Right:
        {
            glm::vec2 right(1.0f, 0.0f);
            right = glm::rotate(right, currentState.angle);
            right *= sideThrust;
            l_velocity += right;
            l_angularVel += rotationForce;
            break;
        }
        default:
            break;
    }

    currentState.angle += l_angle * dt;
    currentState.x += l_pos.x * dt;
    currentState.y += l_pos.y * dt;

    currentState.vx += l_velocity.x * dt;
    currentState.vy += l_velocity.y * dt;

    currentState.vx *= velocityFriction;
    currentState.vy *= velocityFriction;

    currentState.angularVelocity += l_angularVel * dt;
    currentState.angularVelocity *= angularFriction;

    glm::vec2 L1(-size.x - size.z, -size.y - size.z), L2(size.x + size.z, -size.y - size.z);

    L1 = glm::rotate(L1, currentState.angle);
    L2 = glm::rotate(L2, currentState.angle);

    L1 += glm::vec2(currentState.x, currentState.y), L2 += glm::vec2(currentState.x, currentState.y);

    if (L1.y < platformHeight)
        currentState.leg_1_land = true;
    else
        currentState.leg_1_land = false;

    if (L2.y < platformHeight)
        currentState.leg_2_land = true;
    else
        currentState.leg_2_land = false;
}

Step_return Lunar_Lander::step(Action action)
{
    Step_return r_step;

    r_step.state = squizzForNetwork(currentState);

    update(0.03f, action);
    r_step.reward = 0.0f;

    r_step.next_state = squizzForNetwork(currentState);

    r_step.action = action;
    r_step.terminated = crashed() || landed() || !inBounds(50.0f);
    bool _landed = inBoundsLand(0.0f);
    bool _crashed = r_step.terminated && !_landed;
    r_step.truncated = false;
    bool done = r_step.terminated;

    float shaping =
        getShaping(x_shapingValue - r_step.next_state.x, y_shapingValue - r_step.next_state.y, r_step.next_state.x,
                   r_step.next_state.y, r_step.next_state.angle, r_step.next_state.leg_1_land, r_step.next_state.leg_2_land);

    if (prev_shaping != 100000.0f) r_step.reward += (shaping - prev_shaping);
    prev_shaping = shaping;

    switch (action)
    {
        case 1:
        {
            r_step.reward -= 0.03f;
            break;
        }
        case 2:
        {
            r_step.reward -= 0.3f;
            break;
        }
        case 3:
        {
            r_step.reward -= 0.03f;
            break;
        }
    }

    if (_crashed)
    {
        r_step.reward = -100.0f;
    }
    if (_landed)
    {
        r_step.reward = 100.0f;
    }
    last_reward = r_step.reward;
    return r_step;
}

bool Lunar_Lander::landed() { return currentState.leg_1_land && currentState.leg_2_land; }

bool Lunar_Lander::inBoundsLand(float range)
{
    glm::vec2 L1(-size.x - size.z, -size.y - size.z), L2(size.x + size.z, -size.y - size.z);
    L1 = glm::rotate(L1, currentState.angle);
    L2 = glm::rotate(L2, currentState.angle);

    L1 += glm::vec2(currentState.x, currentState.y), L2 += glm::vec2(currentState.x, currentState.y);

    return (currentState.leg_1_land && currentState.leg_2_land) && (L1.x >= land_1 - range && L2.x <= land_2 + range);
}

bool Lunar_Lander::inBounds(float range)
{
    return currentState.x >= land_1 - range && currentState.x <= land_2 + range && currentState.y <= skyLimit;
}

bool Lunar_Lander::crashed()
{
    glm::vec2 A(-size.x, size.y), B(size.x, size.y), C(-size.x, -size.y), D(size.x, -size.y);

    A = glm::rotate(A, currentState.angle);
    B = glm::rotate(B, currentState.angle);
    C = glm::rotate(C, currentState.angle);
    D = glm::rotate(D, currentState.angle);

    A += glm::vec2(currentState.x, currentState.y), B += glm::vec2(currentState.x, currentState.y),
        C += glm::vec2(currentState.x, currentState.y), D += glm::vec2(currentState.x, currentState.y);
    return A.y <= platformHeight || B.y <= platformHeight || C.y <= platformHeight || D.y <= platformHeight;
}

float Lunar_Lander::getShaping(float x, float y, float vx, float vy, float angle, bool leg1, bool leg2)
{
    return 
        (-100.0f * sqrt(x * x + y * y) - 100.0f * sqrt(vx * vx + vy * vy) - 100.0f * abs(angle) + 10.0f * leg1 + 10.0f * leg2);
}

State Lunar_Lander::squizzForNetwork(State state)
{
    state.x /= windowSize.x;
    state.y /= windowSize.y;
    state.vx /= windowSize.x;
    state.vy /= windowSize.y;
    return state;
}

void Lunar_Lander::setWindowSize(glm::vec2 size) { windowSize = size; }

State Lunar_Lander::getState() { return currentState; }

Float_State Lunar_Lander::StateToFloat_State(State state)
{
    Float_State new_state;
    new_state.state.push_back(state.x);
    new_state.state.push_back(state.y);
    new_state.state.push_back(state.vx);
    new_state.state.push_back(state.vy);
    new_state.state.push_back(state.angle);
    new_state.state.push_back(state.angularVelocity);
    new_state.state.push_back(static_cast<float>(state.leg_1_land));
    new_state.state.push_back(static_cast<float>(state.leg_2_land));
    return new_state;
}

Float_Step_Return Lunar_Lander::StepReturnToFloatStepReturn(Step_return step_return)
{
    Float_Step_Return floatStepReturn;
    floatStepReturn.state = StateToFloat_State(step_return.state);
    floatStepReturn.next_state = StateToFloat_State(step_return.next_state);
    floatStepReturn.reward = step_return.reward;
    floatStepReturn.action = static_cast<float>(step_return.action);
    floatStepReturn.terminated = static_cast<float>(step_return.terminated);
    floatStepReturn.truncated = static_cast<float>(step_return.truncated);
    return floatStepReturn;
}

```

And this is the track environment that I created

```C++
struct Track 
{
public:
    std::vector<glm::vec2> trackBorder;
    std::vector<glm::vec4> checkpoints;
    glm::vec4 playerStartValues = glm::vec4(0.0f);
};

struct State
{
    float l0;
    float l1;
    float l2;
    float l3;
    float l4;
    float velocity;
};

struct Triangle
{
    glm::vec2 A = glm::vec2(0.0f), B=glm::vec2(0.0f), C=glm::vec2(0.0f);
    int indexes[3] = {-1,-1,-1};
};

enum Action
{
    Nothing,
    TurnLeft,
    Forward,
    TurnRight,
    Break,
    /*ForwardLeft,
    ForwardRight,
    BackwardLeft,
    BackwardRight,*/
};

struct Step_return
{
    State state;
    Action action;
    State next_state;
    float reward;
    bool terminated;
    bool truncated;
};

struct Float_State;
struct Float_Step_Return;
struct Full_Float_Step_Return;

class Racing_Track
{
public:
    
    Racing_Track();
    Racing_Track(glm::vec2 windowSize);

    Float_State StateToFloat_State(State state);
    Float_Step_Return StepReturnToFloatStepReturn(Step_return step_return);
    Full_Float_Step_Return StepReturnToFullFLoatStepReturn(Step_return step_return);

    void reset();
    void updateEnvironment(float dt, Action a);
    bool hitCheckpoint(int checkpoint, glm::vec2 prev_pos, glm::vec2 current_pos);
    State squizzForNetwork(State state);
    Step_return step(float dt, Action action);
    float edgeFunction(glm::vec2 a, glm::vec2 b, glm::vec2 p);
    Triangle insideWhichTraingle(glm::vec2 point);
    bool insideATriangle(glm::vec2 A,glm::vec2 B,glm::vec2 C,glm::vec2 point);
    bool isOnTrack();
    glm::vec3 lineIntersect(glm::vec2 a, glm::vec2 b, glm::vec2 c, glm::vec2 d);
    glm::vec2 trackRaycast(glm::vec2 from,glm::vec2 to);

    void addTrackPoint(glm::vec2 point);
    void removeLastTrackPoint();
    void addCheckpoint(glm::vec4 checkpoint);
    void removeLastCheckpoint();

    void serializeTrack(std::string path);
    void deserializeTrack(std::string path);

    //Mouse input conversion values
    glm::vec2 projectionDimensions;
    glm::vec2 gameWindowPos;
    glm::vec2 gameWindowSize;

    //building variables
    bool seeTrack =true;
    float lineAngle = 0.0f;
    float checkpointSize = 10.0f;
    int mouseWheel = 0;

    //state
    glm::vec2 position;
    int currentCheckpoint = 0;
    int lastCheckpoint = 0;
    bool badReturn = false;
    float angle = 0.0f;
    float angularVelocity = 0.0f;
    bool done = false;
    int insideTriangleFirstCorner = 0;

    //glm::vec2 line_direction=glm::vec2(0.0f,1.0f);
    glm::vec2 l0 = glm::vec2(-1.0f, 0.0f);
    glm::vec2 l1 = glm::vec2(-0.5f, 0.5f);
    glm::vec2 l2 = glm::vec2(0.0f, 1.0f);
    glm::vec2 l3 = glm::vec2(0.5f, 0.5f);
    glm::vec2 l4 = glm::vec2(1.0f, 0.0f);

    glm::vec2 carRotationSize = glm::vec2(0.0f, 1.0f);


    glm::vec2 size = glm::vec2(5.0f, 10.0f);
    //Shouldn't be changed durring the Running or Training mode
    float forwardForce = 100.0f;
    float breakForce = 100.0f;
    float rotationForce = 10.0f;
    float angularFriction = 0.98f;
    float maxVelocity = 40.0f;
    float velocityFriction = 0.999f;
    
    //Rewards values
    float shapingMultiplier = 0.001f;
    float minimumVelocityValueToKeep = 3.0f;
    float underMinVelRValue = -0.1f;
    float minDistanceFromWall = 5.0f;
    float underMinWallDisRValue = -0.2f;
    float forwardPReward = 0.0f;
    float doNothingNReward = -1.0f;
    float useBreakNReward = -0.1f;  //-0.1f;
    float hitCheckpointReward = 10.0f;
    float winReward = 100.0f;
    float dieReward = -100.0f;

    float prev_shaping = 100000.0f;

    State currentState;
    Track track;

    EnvironmentState envState = EnvironmentState::None;
    BuildingMods buildingMod = BuildingMods::Null;
    Player player = Player::User;

    bool runDebug = true;
};
```

```C++
using namespace glm;

Racing_Track::Racing_Track() { reset(); }

Racing_Track::Racing_Track(glm::vec2 windowSize) : projectionDimensions(windowSize) { reset(); }

void Racing_Track::reset()
{
    /*currentState.x = track.playerStartValues.x;
    currentState.y = track.playerStartValues.y;
    currentState.velocity = 0.0f;
    currentState.angle = track.playerStartValues.z;
    currentState.angularVelocity = 0.0f;
    currentState.currentCheckPoint = 0;
    currentState.wallHitX = 0.0f;
    currentState.wallHitY = 0.0f;*/
    prev_shaping = 100000.0f;
    currentState.velocity = 0.0f;
    position = vec2(track.playerStartValues.x, track.playerStartValues.y);
    currentCheckpoint = 0;
    angle = track.playerStartValues.z;
    angularVelocity = 0;

    done = false;
    // currentState.l0 = 100.0f;
    // currentState.l1 = 100.0f;
    // currentState.l2 = 100.0f;
    // currentState.l3 = 100.0f;
    // currentState.l4 = 100.0f;

    float _lenght;
    vec2 _position = position, l_lastOrientation, _size, A, B, C, D;

    A = B = C = D = position;

    _size = glm::rotate(vec2(-size.x, size.y), angle);
    A += vec2(_size.x * track.playerStartValues.w, _size.y * track.playerStartValues.w);
    _size = glm::rotate(vec2(size.x, size.y), angle);
    B += vec2(_size.x * track.playerStartValues.w, _size.y * track.playerStartValues.w);
    _size = glm::rotate(vec2(size.x, -size.y), angle);
    C += vec2(_size.x * track.playerStartValues.w, _size.y * track.playerStartValues.w);
    _size = glm::rotate(vec2(-size.x, -size.y), angle);
    D += vec2(_size.x * track.playerStartValues.w, _size.y * track.playerStartValues.w);

    _size = (A + D) / 2.0f;
    l_lastOrientation = glm::rotate(l0, angle);
    _position = trackRaycast(_size, _size + l_lastOrientation * 1000.0f);
    currentState.l0 = glm::length(_position - _size);

    _size = A;
    l_lastOrientation = glm::rotate(l1, angle);
    _position = trackRaycast(_size, _size + l_lastOrientation * 1000.0f);
    currentState.l1 = glm::length(_position - _size);

    _size = (A + B) / 2.0f;
    l_lastOrientation = glm::rotate(l2, angle);
    _position = trackRaycast(_size, _size + l_lastOrientation * 1000.0f);
    currentState.l2 = glm::length(_position - _size);

    _size = B;
    l_lastOrientation = glm::rotate(l3, angle);
    _position = trackRaycast(_size, _size + l_lastOrientation * 1000.0f);
    currentState.l3 = glm::length(_position - _size);

    _size = (C + B) / 2.0f;
    l_lastOrientation = glm::rotate(l4, angle);
    _position = trackRaycast(_size, _size + l_lastOrientation * 1000.0f);
    currentState.l4 = glm::length(_position - _size);

    insideTriangleFirstCorner = insideWhichTraingle(position).indexes[0];
}

void Racing_Track::updateEnvironment(float dt, Action a)
{
    glm::vec2 l_lastOrientation = glm::rotate(vec2(0.0f, 1.0f), angle);
    glm::vec2 l_pos = l_lastOrientation * currentState.velocity;
    float l_velocity = 0.0f;
    float l_angle = angularVelocity;
    float l_angularVel = 0.0f;

    switch (a)
    {
        case Action::Nothing:
        {
            break;
        }
        case Action::TurnLeft:
        {
            l_angularVel += rotationForce;
            break;
        }
        case Action::Forward:
        {
            l_velocity += forwardForce;
            break;
        }
        case Action::TurnRight:
        {
            l_angularVel -= rotationForce;
            break;
        }
        case Action::Break:
        {
            l_velocity -= breakForce;
            break;
        }
        default:
            break;
    }

    float m0, m1;

    vec2 A, B, C, D, _size;
    vec2 _position = position;
    A = B = vec2(0.0f, 0.0f);

    angle += l_angle * dt;
    position += l_pos * dt;
    badReturn = false;
    if (track.checkpoints.size() > 0)
    {
        bool checkpoint = hitCheckpoint(currentCheckpoint,_position,position);
        if (checkpoint)
        {
            currentCheckpoint++;
        }
    }

    currentState.velocity += l_velocity * dt;

    currentState.velocity *= velocityFriction;

    if (currentState.velocity < 0.0f) currentState.velocity = 0.0f;

    if (currentState.velocity > maxVelocity) currentState.velocity = maxVelocity;

    angularVelocity += l_angularVel * dt;
    angularVelocity *= angularFriction;

    float _lenght;

    A = B = C = D = position;

    _size = glm::rotate(vec2(-size.x, size.y), angle);
    A += vec2(_size.x * track.playerStartValues.w, _size.y * track.playerStartValues.w);
    _size = glm::rotate(vec2(size.x, size.y), angle);
    B += vec2(_size.x * track.playerStartValues.w, _size.y * track.playerStartValues.w);
    _size = glm::rotate(vec2(size.x, -size.y), angle);
    C += vec2(_size.x * track.playerStartValues.w, _size.y * track.playerStartValues.w);
    _size = glm::rotate(vec2(-size.x, -size.y), angle);
    D += vec2(_size.x * track.playerStartValues.w, _size.y * track.playerStartValues.w);

    _size = (A + D) / 2.0f;
    l_lastOrientation = glm::rotate(l0, angle);
    _position = trackRaycast(_size, _size + l_lastOrientation * 1000.0f);
    currentState.l0 = glm::length(_position - _size);

    _size = A;
    l_lastOrientation = glm::rotate(l1, angle);
    _position = trackRaycast(_size, _size + l_lastOrientation * 1000.0f);
    currentState.l1 = glm::length(_position - _size);

    _size = (A + B) / 2.0f;
    l_lastOrientation = glm::rotate(l2, angle);
    _position = trackRaycast(_size, _size + l_lastOrientation * 1000.0f);
    currentState.l2 = glm::length(_position - _size);

    _size = B;
    l_lastOrientation = glm::rotate(l3, angle);
    _position = trackRaycast(_size, _size + l_lastOrientation * 1000.0f);
    currentState.l3 = glm::length(_position - _size);

    _size = (C + B) / 2.0f;
    l_lastOrientation = glm::rotate(l4, angle);
    _position = trackRaycast(_size, _size + l_lastOrientation * 1000.0f);
    currentState.l4 = glm::length(_position - _size);
}

bool Racing_Track::hitCheckpoint(int checkpoint, glm::vec2 prev_pos, glm::vec2 current_pos)
{
    glm::vec2 A, B,_size;
    float m0, m1;

    A = B = vec2(track.checkpoints[currentCheckpoint].x, track.checkpoints[currentCheckpoint].y);
    _size = glm::rotate(l2, track.checkpoints[currentCheckpoint].z);
    A -= track.checkpoints[currentCheckpoint].w * _size;
    B += track.checkpoints[currentCheckpoint].w * _size;
    m0 = edgeFunction(A, B, prev_pos);


    m1 = edgeFunction(A, B, current_pos);
    if ((m1 >= 0.0f && m0 <= 0.0f) || (m0 >= 0.0f && m1 <= 0.0f))
    {
        if (glm::length((vec2(track.checkpoints[currentCheckpoint].x, track.checkpoints[currentCheckpoint].y) -
                         vec2(current_pos))) <= track.checkpoints[currentCheckpoint].w)
        {
            return true;
        }
    }
    return false;
}

State Racing_Track::squizzForNetwork(State state)
{
    return state;
}

Step_return Racing_Track::step(float dt, Action action)
{
    Step_return r_step;
    r_step.state = /*squizzForNetwork(*/ currentState /*)*/;
    lastCheckpoint = currentCheckpoint;
    updateEnvironment(dt, action);

    int currentTriangleFirstCorner = insideWhichTraingle(position).indexes[0];
    r_step.reward = 0.0f;

    r_step.next_state = /*squizzForNetwork(*/ currentState /*)*/;
    r_step.action = action;
    bool notOnTrack = !isOnTrack();
    bool finishedTrack = (currentCheckpoint >= track.checkpoints.size());
    done = r_step.terminated = notOnTrack || finishedTrack;

    r_step.truncated = false;

    float shaping = shapingMultiplier * r_step.next_state.velocity;

    if (prev_shaping != 100000.0f) r_step.reward += (shaping - prev_shaping);

    prev_shaping = shaping;
    insideTriangleFirstCorner = currentTriangleFirstCorner;

    if (r_step.next_state.velocity < minimumVelocityValueToKeep)
    {
        r_step.reward += underMinVelRValue;
    }

    if (r_step.next_state.l0 < minDistanceFromWall || r_step.next_state.l1 < minDistanceFromWall ||
        r_step.next_state.l2 < minDistanceFromWall || r_step.next_state.l3 < minDistanceFromWall ||
        r_step.next_state.l4 < minDistanceFromWall)
    {
        r_step.reward += underMinWallDisRValue;
    }

    switch (action)
    {
        case Action::Forward:
        {
            r_step.reward += forwardPReward;
            break;
        }
        case Action::Nothing:
        {
            r_step.reward += doNothingNReward;
            break;
        }
        case Action::Break:
        {
            r_step.reward += useBreakNReward;
            break;
        }
        default:
        {
            break;
        }
    }
    if (notOnTrack)
    {
        r_step.reward = dieReward;
    }
    if (currentCheckpoint != lastCheckpoint)
    {
        r_step.reward += hitCheckpointReward;
    }
    if (finishedTrack)
    {
        r_step.reward = winReward;
    }

    return r_step;
}

Float_State Racing_Track::StateToFloat_State(State state)
{
    Float_State new_state;
    /*new_state.state.push_back(state.x);
    new_state.state.push_back(state.y);
    new_state.state.push_back(state.velocity);
    new_state.state.push_back(state.angle);
    new_state.state.push_back(state.angularVelocity);
    new_state.state.push_back(state.currentCheckPoint);
    new_state.state.push_back(state.wallHitX);
    new_state.state.push_back(state.wallHitY);*/
    new_state.state.push_back(state.l0);
    new_state.state.push_back(state.l1);
    new_state.state.push_back(state.l2);
    new_state.state.push_back(state.l3);
    new_state.state.push_back(state.l4);
    new_state.state.push_back(state.velocity);
    return new_state;
}

Float_Step_Return Racing_Track::StepReturnToFloatStepReturn(Step_return step_return)
{
    Float_Step_Return floatStepReturn;
    floatStepReturn.state = StateToFloat_State(step_return.state);
    floatStepReturn.next_state = StateToFloat_State(step_return.next_state);
    floatStepReturn.reward = step_return.reward;
    floatStepReturn.action = static_cast<float>(step_return.action);
    floatStepReturn.terminated = static_cast<float>(step_return.terminated);
    return floatStepReturn;
}

Full_Float_Step_Return Racing_Track::StepReturnToFullFLoatStepReturn(Step_return step_return)
{
    Full_Float_Step_Return floatStepReturn;
    // state
    floatStepReturn.data.push_back(step_return.state.l0);
    floatStepReturn.data.push_back(step_return.state.l1);
    floatStepReturn.data.push_back(step_return.state.l2);
    floatStepReturn.data.push_back(step_return.state.l3);
    floatStepReturn.data.push_back(step_return.state.l4);
    floatStepReturn.data.push_back(step_return.state.velocity);
    // new state
    floatStepReturn.data.push_back(step_return.next_state.l0);
    floatStepReturn.data.push_back(step_return.next_state.l1);
    floatStepReturn.data.push_back(step_return.next_state.l2);
    floatStepReturn.data.push_back(step_return.next_state.l3);
    floatStepReturn.data.push_back(step_return.next_state.l4);
    floatStepReturn.data.push_back(step_return.next_state.velocity);
    // reward
    floatStepReturn.data.push_back(step_return.reward);
    // action
    floatStepReturn.data.push_back(static_cast<float>(step_return.action));
    // terminated
    floatStepReturn.data.push_back(static_cast<float>(step_return.terminated));
    return floatStepReturn;
}

float Racing_Track::edgeFunction(glm::vec2 a, glm::vec2 b, glm::vec2 p)
{
    return (p.x - a.x) * (b.y - a.y) - (p.y - a.y) * (b.x - a.x);
}

Triangle Racing_Track::insideWhichTraingle(glm::vec2 point)
{
    Triangle triangle;

    if (track.trackBorder.size() > 2)
    {
        vec2 A, B, C;
        for (int i = 0; i < track.trackBorder.size() - 2; i++)
        {
            A = track.trackBorder[i];
            B = track.trackBorder[i + 1];
            C = track.trackBorder[i + 2];

            if (insideATriangle(A, B, C, point))
            {
                triangle.A = A;
                triangle.B = B;
                triangle.C = C;
                triangle.indexes[0] = i;
                triangle.indexes[1] = i + 1;
                triangle.indexes[2] = i + 2;
            }
        }

        A = track.trackBorder[track.trackBorder.size() - 1];
        B = track.trackBorder[track.trackBorder.size() - 2];
        C = track.trackBorder[0];
        if (insideATriangle(A, B, C, point))
        {
            triangle.A = B;
            triangle.B = A;
            triangle.C = C;
            triangle.indexes[0] = track.trackBorder.size() - 2;
            triangle.indexes[1] = track.trackBorder.size() - 1;
            triangle.indexes[2] = 0;
        }

        A = track.trackBorder[track.trackBorder.size() - 2];
        B = track.trackBorder[1];
        C = track.trackBorder[0];
        if (insideATriangle(A, B, C, point))
        {
            triangle.A = A;
            triangle.B = B;
            triangle.C = C;
            triangle.indexes[0] = track.trackBorder.size() - 2;
            triangle.indexes[1] = 1;
            triangle.indexes[2] = 0;
        }
    }
    return triangle;
}

bool Racing_Track::insideATriangle(glm::vec2 A, glm::vec2 B, glm::vec2 C, glm::vec2 point)
{
    float m0, m1, m2;
    m0 = edgeFunction(A, B, point);
    m1 = edgeFunction(B, C, point);
    m2 = edgeFunction(C, A, point);
    if ((m0 >= 0.0f && m1 >= 0.0f && m2 >= 0.0f) || (m0 < 0.0f && m1 < 0.0f && m2 < 0.0f)) return true;
    return false;
}

bool Racing_Track::isOnTrack()
{
    vec2 A, B, C, D;
    A = B = C = D = position;
    vec2 l_size;

    l_size = glm::rotate(vec2(-size.x, size.y), angle);
    A += vec2(l_size.x * track.playerStartValues.w, l_size.y * track.playerStartValues.w);
    l_size = glm::rotate(vec2(size.x, size.y), angle);
    B += vec2(l_size.x * track.playerStartValues.w, l_size.y * track.playerStartValues.w);
    l_size = glm::rotate(vec2(size.x, -size.y), angle);
    C += vec2(l_size.x * track.playerStartValues.w, l_size.y * track.playerStartValues.w);
    l_size = glm::rotate(vec2(-size.x, -size.y), angle);
    D += vec2(l_size.x * track.playerStartValues.w, l_size.y * track.playerStartValues.w);

    Triangle triangle = insideWhichTraingle(A);

    if (triangle.indexes[0] == -1) return false;

    triangle = insideWhichTraingle(B);
    if (triangle.indexes[0] == -1) return false;

    triangle = insideWhichTraingle(C);
    if (triangle.indexes[0] == -1) return false;

    triangle = insideWhichTraingle(D);
    if (triangle.indexes[0] == -1) return false;

    return true;
}

glm::vec3 Racing_Track::lineIntersect(glm::vec2 a, glm::vec2 b, glm::vec2 c, glm::vec2 d)
{
    vec2 r = b - a;
    vec2 s = d - c;
    float _d = r.x * s.y - r.y * s.x;
    float u = ((c.x - a.x) * r.y - (c.y - a.y) * r.x) / _d;
    float t = ((c.x - a.x) * s.y - (c.y - a.y) * s.x) / _d;
    bool intersect = (0.0f <= u && u <= 1.0f && 0.0f <= t && t <= 1.0f);
    r = a + t * r;
    return vec3(r.x, r.y, static_cast<float>(intersect));
}

glm::vec2 Racing_Track::trackRaycast(glm::vec2 from, glm::vec2 to)
{
    vec2 intersectionPoint = vec2(0.0f);
    float lenght = 10000, _lenght;
    // glm::vec4 color(1.0f, 1.0f, 1.0f, 1.0f);
    // bee::Engine.DebugRenderer().AddLine(bee::DebugCategory::General, from, to, color);
    // color = vec4(1.0f, 0.0f, 0.0f, 1.0f);
    if (track.trackBorder.size() > 2)
    {
        vec2 A, C;
        vec3 result;
        for (int i = 0; i < track.trackBorder.size() - 2; i++)
        {
            A = track.trackBorder[i];
            C = track.trackBorder[i + 2];
            result = lineIntersect(from, to, A, C);
            if (static_cast<bool>(result.z))
            {
                _lenght = glm::length(vec2(result.x, result.y) - from);
                if (_lenght < lenght)
                {
                    lenght = _lenght;
                    intersectionPoint = vec2(result.x, result.y);
                }
            }
        }

        A = track.trackBorder[0];
        C = track.trackBorder[track.trackBorder.size() - 1];
        result = lineIntersect(from, to, A, C);
        if (static_cast<bool>(result.z))
        {
            _lenght = glm::length(vec2(result.x, result.y) - from);
            if (_lenght < lenght)
            {
                lenght = _lenght;
                intersectionPoint = vec2(result.x, result.y);
            }
        }

        A = track.trackBorder[1];
        C = track.trackBorder[track.trackBorder.size() - 2];
        result = lineIntersect(from, to, A, C);
        if (static_cast<bool>(result.z))
        {
            _lenght = glm::length(vec2(result.x, result.y) - from);
            if (_lenght < lenght)
            {
                lenght = _lenght;
                intersectionPoint = vec2(result.x, result.y);
            }
        }
    }

    return intersectionPoint;
}

void Racing_Track::addTrackPoint(glm::vec2 point) { track.trackBorder.push_back(point); }

void Racing_Track::removeLastTrackPoint()
{
    if (!track.trackBorder.empty()) track.trackBorder.pop_back();
}

void Racing_Track::addCheckpoint(glm::vec4 checkpoint) { track.checkpoints.push_back(checkpoint); }

void Racing_Track::removeLastCheckpoint()
{
    if (!track.checkpoints.empty()) track.checkpoints.pop_back();
}

void Racing_Track::serializeTrack(std::string path)
{
    std::string data = " ";
    for (auto& point : track.trackBorder)
    {
        data += std::to_string(point.x) + " " + std::to_string(point.y) + " ";
    }
    bee::Engine.FileIO().WriteTextFile(bee::FileIO::Directory::Asset, "Tracks/" + path + "_track.txt", data);

    data = " ";
    for (auto& point : track.checkpoints)
    {
        data += std::to_string(point.x) + " " + std::to_string(point.y) + " " + std::to_string(point.z) + " " +
                std::to_string(point.w) + " ";
    }
    data += std::to_string(track.playerStartValues.x) + " " + std::to_string(track.playerStartValues.y) + " " +
            std::to_string(track.playerStartValues.z) + " " + std::to_string(track.playerStartValues.w) + " ";
    bee::Engine.FileIO().WriteTextFile(bee::FileIO::Directory::Asset, "Tracks/" + path + "_checkpoints.txt", data);
}

void Racing_Track::deserializeTrack(std::string path)
{
    track.trackBorder.clear();
    track.checkpoints.clear();

    std::string string_data;
    std::string value;
    std::vector<float> float_data;
    float x, y, z, w;
    std::string l_path = "Tracks/" + path + "_track.txt";
    string_data = bee::Engine.FileIO().ReadTextFile(bee::FileIO::Directory::Asset, l_path);
    while (string_data.size() > 1)
    {
        size_t initialPos = string_data.find(" ");
        size_t finalPos = string_data.find(" ", 1);
        value = string_data.substr(initialPos, finalPos - initialPos);
        x = std::stof(value);
        string_data = string_data.substr(finalPos, string_data.size() - finalPos);

        initialPos = string_data.find(" ");
        finalPos = string_data.find(" ", 1);
        value = string_data.substr(initialPos, finalPos - initialPos);
        y = std::stof(value);
        string_data = string_data.substr(finalPos, string_data.size() - finalPos);
        track.trackBorder.push_back(glm::vec2(x, y));
    }

    float_data.clear();

    l_path = "Tracks/" + path + "_checkpoints.txt";
    string_data = bee::Engine.FileIO().ReadTextFile(bee::FileIO::Directory::Asset, l_path);
    while (string_data.size() > 1)
    {
        size_t initialPos = string_data.find(" ");
        size_t finalPos = string_data.find(" ", 1);
        value = string_data.substr(initialPos, finalPos - initialPos);
        x = std::stof(value);
        string_data = string_data.substr(finalPos, string_data.size() - finalPos);

        initialPos = string_data.find(" ");
        finalPos = string_data.find(" ", 1);
        value = string_data.substr(initialPos, finalPos - initialPos);
        y = std::stof(value);
        string_data = string_data.substr(finalPos, string_data.size() - finalPos);

        initialPos = string_data.find(" ");
        finalPos = string_data.find(" ", 1);
        value = string_data.substr(initialPos, finalPos - initialPos);
        z = std::stof(value);
        string_data = string_data.substr(finalPos, string_data.size() - finalPos);

        initialPos = string_data.find(" ");
        finalPos = string_data.find(" ", 1);
        value = string_data.substr(initialPos, finalPos - initialPos);
        w = std::stof(value);
        string_data = string_data.substr(finalPos, string_data.size() - finalPos);

        track.checkpoints.push_back(glm::vec4(x, y, z, w));
    }
    if (track.checkpoints.size() >= 1)
    {
        track.playerStartValues = track.checkpoints[track.checkpoints.size() - 1];
        track.checkpoints.pop_back();

        carRotationSize.y = track.playerStartValues.w;
        carRotationSize.x = track.playerStartValues.z;
    }
}

```
