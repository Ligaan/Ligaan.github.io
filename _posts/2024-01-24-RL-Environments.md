## Disclaimer, the content used in this blog is part of a project made for buas (Breda University of Applied Science)

Now that we are done with the DQN, it is time to discuss a bit about environments.

When training an agent in an environment with RL algorithms, one of the key points in determining whether or not it learns how to solve the environment is the reward system. A good reward system could stimulate the agent to learn better and faster, while a bad reward system could make the agent fail to learn something useful. This shows some of the [gym environments](https://github.com/openai/gym/tree/master/gym/envs) which might be a good place to look for examples.

First, those are the includes that we will use
```cpp
#include <glm/glm.hpp>
#include <vector>
#include <string>
#include "glm/gtx/rotate_vector.hpp"
#include "cmath"
```

Now, let's start by creating some structs that we will use later.

This will store all the information that you will need for loading/saving a track.
```C++
struct Track 
{
public:
    std::vector<glm::vec2> trackBorder;
    std::vector<glm::vec4> checkpoints;
    glm::vec4 playerStartValues = glm::vec4(0.0f);
};
```
This is our observation space, and the values are as follows:
- l0 is a sensor on the left side of the car that tells it how far it is from a wall.
- l1 is a sensor on the front left corner of the car that tells it how far it is from a wall.
- l2 is a sensor on the front side of the car that tells it how far it is from a wall
- l3 is a sensor on the front right side of the car that tells it how far it is from a wall.
- l4 is a sensor on the right side of the car that tells it how far it is from a wall.
- velocity is the current velocity of the car.
```cpp
struct State
{
    float l0;
    float l1;
    float l2;
    float l3;
    float l4;
    float velocity;
};
```
This struct will be used when we check if the car is on the track.
```cpp
struct Triangle
{
    glm::vec2 A = glm::vec2(0.0f), B=glm::vec2(0.0f), C=glm::vec2(0.0f);
    int indexes[3] = {-1,-1,-1};
};
```
This is the action space of the environment.
```cpp
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
```
As in the previous example this is the step return struct
```cpp
struct Step_return
{
    State state;
    Action action;
    State next_state;
    float reward;
    bool terminated;
    bool truncated;
};
```
Now that we have those, it is time to build the track.
We will start with a header that looks like this.
```cpp
struct Float_State;
struct Float_Step_Return;
struct Full_Float_Step_Return;
class Racing_Track
{
public:
    
    Racing_Track();

    //This will give us the camera space size
    Racing_Track(glm::vec2 windowSize);

    // Some functions for conversion into data accessible by the DQN class
    Float_State StateToFloat_State(State state);
    Float_Step_Return StepReturnToFloatStepReturn(Step_return step_return);
    Full_Float_Step_Return StepReturnToFullFLoatStepReturn(Step_return step_return);

    // Resets the environment
    void reset();
    void updateEnvironment(float dt, Action a);

    // Check if the agent went past the checkpoint in between the current state and the previous state
    bool hitCheckpoint(int checkpoint, glm::vec2 prev_pos, glm::vec2 current_pos);

    Step_return step(float dt, Action action);

    // This function will be used to determine if a point is on the left or right side of a line
    float edgeFunction(glm::vec2 a, glm::vec2 b, glm::vec2 p);

    // This will give us the triangle in which the point is contained
    Triangle insideWhichTraingle(glm::vec2 point);

    // This checks to see if a point is within a triangle
    bool insideATriangle(glm::vec2 A,glm::vec2 B,glm::vec2 C,glm::vec2 point);

    bool isOnTrack();

    // This is for the raycasting that we will do
    glm::vec3 lineIntersect(glm::vec2 a, glm::vec2 b, glm::vec2 c, glm::vec2 d);
    glm::vec2 trackRaycast(glm::vec2 from,glm::vec2 to);

    // This is for building the track, since most of the time we want to train the agent with multiple environments in order to avoid scenarios where it overspecializes on a specific environment.
    void addTrackPoint(glm::vec2 point);
    void removeLastTrackPoint();
    void addCheckpoint(glm::vec4 checkpoint);
    void removeLastCheckpoint();

    //Mouse input conversion values
    glm::vec2 projectionDimensions;
    glm::vec2 gameWindowPos;
    glm::vec2 gameWindowSize;

    //building variables
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

    glm::vec2 l0 = glm::vec2(-1.0f, 0.0f);
    glm::vec2 l1 = glm::vec2(-0.5f, 0.5f);
    glm::vec2 l2 = glm::vec2(0.0f, 1.0f);
    glm::vec2 l3 = glm::vec2(0.5f, 0.5f);
    glm::vec2 l4 = glm::vec2(1.0f, 0.0f);

    glm::vec2 carRotationSize = glm::vec2(0.0f, 1.0f);


    glm::vec2 size = glm::vec2(5.0f, 10.0f);

    //Shouldn't be changed during the Run or Train mode.
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
    float useBreakNReward = -0.1f; 
    float hitCheckpointReward = 10.0f;
    float winReward = 100.0f;
    float dieReward = -100.0f;

    float prev_shaping = 100000.0f;

    State currentState;
    Track track;
};
```
Now that we have the class header, I'll start by going through the auxiliar functions first.

Those are for converting the state and step return struct data into something that the DQN could use
```cpp
Float_State Racing_Track::StateToFloat_State(State state)
{
    Float_State new_state;
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

```


The track building functions
```cpp
void Racing_Track::removeLastTrackPoint()
{
    if (!track.trackBorder.empty()) track.trackBorder.pop_back();
}

void Racing_Track::addCheckpoint(glm::vec4 checkpoint) { track.checkpoints.push_back(checkpoint); }

void Racing_Track::removeLastCheckpoint()
{
    if (!track.checkpoints.empty()) track.checkpoints.pop_back();
}

void Racing_Track::addTrackPoint(glm::vec2 point) { track.trackBorder.push_back(point); }
```

Those are for checking if and where the car is on the track.
```cpp
float Racing_Track::edgeFunction(glm::vec2 a, glm::vec2 b, glm::vec2 p)
{
    return (p.x - a.x) * (b.y - a.y) - (p.y - a.y) * (b.x - a.x);
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
```
Those functions are for raycasting the lines used for the car sensors.
```cpp
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
```
This function will check to see if the car is past a checkpoint

```cpp
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
```

This is how the class constructors and the reset function look
```cpp
Racing_Track::Racing_Track() { reset(); }

Racing_Track::Racing_Track(glm::vec2 windowSize) : projectionDimensions(windowSize) { reset(); }

void Racing_Track::reset()
{
    prev_shaping = 100000.0f;
    currentState.velocity = 0.0f;
    position = vec2(track.playerStartValues.x, track.playerStartValues.y);
    currentCheckpoint = 0;
    angle = track.playerStartValues.z;
    angularVelocity = 0;

    done = false;

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
```
This is how the update function for the environment looks
```cpp
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

```
This is how the reward system looks. The approach that I went for is that the possitive rewards(checkpoints) are scarce and used to guide the agent toward the direction that it needs to follow, the shaping is used in order to help the agent improve and learn how to go faster and faster, the switch case is used to reinforce the use of certain actions, like for example I encourage the agent to use the forward action and disencourage it from using the breaks or doing nothing, the coditions for the sensors is in order to make the agent keep a certain distance from the track wall in order to avoid accidents and the minimum speed condition is an extra condition for speeding up the process of teaching it that is should always aim for speed. The final conditions are a massive positive reward for finishing the track and a massive negative reward for hitting the track walls.
```cpp
Step_return Racing_Track::step(float dt, Action action)
{
    Step_return r_step;
    r_step.state = currentState;
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
```

Now that we are done with the environment, if we combine the RL algorithm with this environment we should get something like this.

![video](/Images/TrainedModel.gif)
![video](/Images/TrainedModel2.gif)
