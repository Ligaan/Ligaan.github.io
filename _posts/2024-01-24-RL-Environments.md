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
