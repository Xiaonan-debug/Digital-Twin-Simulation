# Revised 5.8.23 Brassil, Functional Circulation v6

# Structure of the analysis Revised 5.8.23:

# Action Vector, 7D, [0 to 6]. This is a single number encoding 8 ternary components (0 to 3 recoded at Step() to -1 to 1)
# 0. Temperature C
# 1. Gas Flow lpm
# 2. Gas Richness %
# 3. Glucose mM
# 4. Insulin mU
# 5. Bicarb mM
# 6. Vasodilator (mL concentration TBD)
# To encode the initial action value as 8 No-Actions, calculate the base-3 number 11111111 as a decimal

# State Vector (new 5.1.23), 6D [0 to 5]. This is a single number encoding 6 ternary components (also 0 to 3 recoded at Step() to -1 to 1)
# 0. Temperature C
# 1. PFI
# 2. pH
# 3. pvO2 mmHg
# 4. Glucose mM
# 5. Insulin mU
# To encode the initial state value as 6 Normal-States, calculate the base-3 number 111111 as a decimal

# For 7 actions and 6 states, the Q-matrix will have total size = 1,594,323 . . . 1.6 million


# Reward is based on 2 factors: PFI and hours, scale 0-255. Hours can be 0-24, and PFI can be 0 to 4.
# Scoring algoorithm 5.5.23: allocate 47% to hours int (multiply hours by 5: 0-120, and allocate remainder to PFI: 33.75 * PFI)
# This is a simple linearized scoring method
# Do this scoring in the StepSimulator() function

import numpy as np 
import gym
import random
from gym import Env, spaces
import time
from FunctionalStepSimulator import SingleStep
import matplotlib.pyplot as plt


# Setup initial values as globals
# State elements initialized
temperatureCelsius = 34
pressuremmHg = 80
perfusionFlowmLpm = 80 # Set VR to 1.0 (i.e., P = F)
vascularResistance = pressuremmHg / perfusionFlowmLpm
perfusionFlowIndex = 100 / (vascularResistance * 300) # Hard code the graft grams  
pH = 7.4
pO2mmHg = 100
pvO2 = 37
svO2 = .65
pvCO2mmHg = 30    
glucosemMolar = 6
insulinmUnits = 50 # Was 25. Set it high enough that it doesnt easily run out
pCO2mmHg = 38
lactatemMolar = 1
hematocrit = 13.75

#bigState addends
bicarbMmoles = 8 #Set to balance-out the 5% CO2 level from the 95-5 respirator gas
gasFlowLPM = .8
gasRichness = .8
hours = 0

# Initialize and encode action value from components to command "No Initial Action"
actionValue = 0 # Seed at 0
possibleActions = 7
noActionValue = 1
for xx in range (0, possibleActions):
    actionValue = actionValue + pow(3,xx) * noActionValue
resetAction = actionValue # Setup the initial action value also to be used at reset
    
# Initialize and encode state value from components to indicale "Normal Initial State"
stateValue = 0 # Seed at 0
possibleStates = 6
normalState = 1
for yy in range (0, possibleStates):
    stateValue = stateValue + pow(3,yy) * normalState
resetState = stateValue # Set up the initial state also to be used at reset

# Initialize the bigState, a list of system states from which the stateValue is calculated, and providing a running practical status of the prep
initialBigState = [temperatureCelsius, pressuremmHg, perfusionFlowmLpm, perfusionFlowIndex, pH, pO2mmHg, pvO2, svO2, pvCO2mmHg, glucosemMolar, insulinmUnits, lactatemMolar, hematocrit, bicarbMmoles, gasFlowLPM, gasRichness, hours]
bigState = initialBigState # Begin bigState to the above initial values at program start
reward = 33.75 # This equates to PFI = 1 and Hours = 0 as an initial value. Will change in future revs when a scoring function is developed
resetReward = reward
stateVector= [0] * possibleStates
reward = 0

# Gym objects and functions
class PumpScape(Env):
    def __init__(self):
        super(PumpScape, self).__init__()
        
        # Define Action and Observation spaces
        self.observation_space = spaces.Discrete(pow(3,possibleStates))
        self.action_space = spaces.Discrete(pow(3,possibleActions))
   
    # Define the step response. The parameter actionCombo is a 2-component list comprising actionValue, bigState which are passed to the step simulator
    def step(self, actionCombo, train=True):
        done = False

        # ----- 3D. CALL (EXTERNAL) SIMULATOR: SEND ACTIONVALUE AND BIGSTATE, GETTING BACK NEW BIGSTATE, STATEVECTOR, and REWARD ----- 
        # Send actionValue and bigState to the simulator
        answer = SingleStep(actionCombo) # Call Single Step() from imported StepSimulator which returns a 3-element list [list, list, float]

        # Here is where we would put the call to the serial port to run the physical system

        # ----- 3E. CALCULATE NEW STATEVALUE, REWARD, DONE, AND BIGSTATE -----
        bigState = answer[0]
        stateVector = answer[1] # stateVector is the list of states R:-1 to 1 
        reward = answer[2] # coreValue is the reward

        stateValue = 0 # stateValue is the stateVector encoded as a single integer (R:0 to 729 for a 6D space)
        
        counts = 0
        for z in range(0, len(stateVector)):
            if stateVector[z] > 1 or stateVector[z] < -1:
                stateVector[z] = 1 * np.sign(stateVector[z])
                if not train:
                    done = True
                counts += 1
            stateValue = stateValue + pow(3,z)*(stateVector[z] + 1) # encode the state vector as a number, also converting to all-positive by adding 1

        # if counts >= 3 and not train:
        #     done = True

        if bigState[16] >= 24:
            done = True
        
        # Returns the obs, reward, done, and bigState. bigState occupies the info field
        # bigState is cycled back into the step as the ongoing practical status
        return stateValue, reward, done, bigState
        
    # Define the reset conditions using all globals (so they should work in global namespace)
    def reset(self):    

        # Reset conditions to initial
        actionValue = resetAction
        stateValue = resetState
        bigState = initialBigState

        # Introduce a stochastic starting vascular resistance that ranges from 0.5 to 2 mmHg/mL/min with a mode of 1.0
        # vascularResistance = random.uniform(0.8, 1.2)
        # perfusionFlowmLpm = pressuremmHg / vascularResistance
        perfusionFlowIndex = .3333
        bigState[2] = perfusionFlowmLpm
        bigState[3] = perfusionFlowIndex
              
        reward = resetReward
        stateVector = [0] * possibleStates 
        return bigState

# Create the Q-learning agent
class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.5):
        self.env = env
        self.alpha = alpha # Learning rate
        self.gamma = gamma # Discount factor
        self.epsilon = epsilon # Epsilon-greedy exploration probability
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n), dtype = np.float32) # Q-table

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:

            return self.env.action_space.sample() # Choose a random action with epsilon probability    
        else:
            return np.argmax(self.q_table[state]) # Choose the action with the highest Q-value for the current state

    def update_q_table(self, state, action, reward, next_state):
        q_max = np.max(self.q_table[next_state]) # Get the maximum Q-value for the next state
        self.q_table[state, action] += self.alpha * (reward + self.gamma * q_max - self.q_table[state, action]) # Update Q-value


####################### Execution begins here:

      
# ----- 1. INSTANTIATE THE CLASS -----
env = PumpScape() # Instantiate the PumpScape() class called env
num_episodes = 15000 # Number of episodes for training
max_steps_per_episode = 24 # Maximum number of steps per episode

# ----- 2. SETUP THE AGENT -----
# Create the Q-learning agent
agent = QLearningAgent(env)
exploration_rate_history = []
# ----- 3. TRAIN -----
# Training loop
for episode in range(num_episodes):

    # ----- 3A. RESET -----
    bigState = env.reset() # Reset the environment and get the initial state
    stateValue = resetState
    
    # ----- 3B. SIMULATE USING HOURLY STEPS -----
    for step in range(max_steps_per_episode):
        actionValue = agent.choose_action(stateValue) # Agent choose an action from R:0 to 6561 either at random or from Q-matrix based on 729 states

        # Update the parameter to send to env.step() comprising  actionValue and bigState
        action = [actionValue, bigState]
        
        # ----- 3C. CALL ENV.STEP(): GET BACK NEXTSTATE, REWARD, DONE, BIGSTATE ----
        next_state, reward, done, bigState = env.step(action) # Take a step in the environment

        # ----- 3F. UPDATE Q-TABLE -----
        agent.update_q_table(stateValue, actionValue, reward, next_state) # Update the Q-table
        stateValue = next_state # Update the current state
        if done: # If the episode is finished
            break
    
    nonzero_count = np.count_nonzero(agent.q_table)
    total_entries = agent.q_table.size
    exploration_rate = (nonzero_count / total_entries) * 100  # percentage
    exploration_rate_history.append(exploration_rate)

# ----- 4. EVALUATE -----
# Evaluation loop
num_eval_episodes = 300 # Number of episodes for evaluation
total_rewards = 0
steps = []

final_bigStates = [] 

for episode in range(num_eval_episodes):
    step_count = 0
    bigState = env.reset() # Reset the environment and get the initial state
    
    for step in range(max_steps_per_episode):
        step_count += 1
        actionValue = np.argmax(agent.q_table[stateValue]) # Choose the action with the highest Q-value for the current state
        action = [actionValue, bigState] # action contains 2 lists: the action vector, and bigState the practical values vector        
        next_state, reward, done, bigState = env.step(action, train=False) # Take a step in the environment
        stateValue = next_state # Update the current state
        total_rewards += reward # Accumulate the total reward

        if done: # If the episode is finished
            break
    steps.append(step_count)
    final_bigStates.append(bigState)

avg_steps = np.mean(steps)
print("Average steps per episode: ", avg_steps)

# Calculate average reward per episode during evaluation
avg_reward_per_episode = total_rewards / num_eval_episodes
print("Average reward per episode during evaluation: ", avg_reward_per_episode)



env.close()

plt.figure(figsize=(12, 8))
plt.imshow(agent.q_table, aspect='auto', cmap='viridis')
plt.colorbar(label='Q-value')
plt.xlabel("Action index")
plt.ylabel("State index")
plt.title("Q-table Heatmap")
plt.savefig('Qtable.png')

plt.figure(figsize=(12, 8))
plt.plot(exploration_rate_history)
plt.xlabel("Episode")
plt.ylabel("Exploration Rate (%)")
plt.title("Exploration Rate Over Episodes")
plt.savefig("exploration_rate.png")
plt.show()

final_bigStates_arr = np.array(final_bigStates)
mean_values = np.mean(final_bigStates_arr, axis=0)
std_values = np.std(final_bigStates_arr, axis=0)

# Parameter names for display (in the same order as defined in initialBigState)
param_names = ["Temperature (C)", "Pressure (mmHg)", "Perfusion Flow (mLpm)",
               "Perfusion Flow Index", "pH", "pO2 (mmHg)", "pvO2", "svO2",
               "pvCO2 (mmHg)", "Glucose (mM)", "Insulin (mU)", "Lactate (mM)",
               "Hematocrit", "Bicarb (mmoles)", "Gas Flow (LPM)", "Gas Richness", "Hours"]

plt.figure(figsize=(14, 8))
bars = plt.bar(range(len(mean_values)), mean_values, yerr=std_values, capsize=5)
plt.bar(range(len(mean_values)), mean_values, yerr=std_values, capsize=5)
plt.xticks(range(len(mean_values)), param_names, rotation=45, ha="right")
plt.ylabel("Average Final Value")
plt.title("Average Final System State Parameters After Evaluation")

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig("final_state_parameters.png")
plt.show()
