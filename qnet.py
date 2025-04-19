
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import time
import os
from gym import Env, spaces
from FunctionalStepSimulator import SingleStep


# seed = 3407
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)

# Check if CUDA is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create output directory for results
output_dir = os.path.expanduser("~/Desktop/Simulator")
os.makedirs(output_dir, exist_ok=True)

# Setup initial values
# State elements initialized
temperatureCelsius = 34
pressuremmHg = 80
perfusionFlowmLpm = 80  # Set VR to 1.0 (i.e., P = F)
vascularResistance = pressuremmHg / perfusionFlowmLpm
perfusionFlowIndex = 100 / (vascularResistance * 300)  # Hard code the graft grams  
pH = 7.4
pO2mmHg = 100
pvO2 = 37
svO2 = .65
pvCO2mmHg = 30    
glucosemMolar = 6
insulinmUnits = 50  # Was 25. Set it high enough that it doesn't easily run out
pCO2mmHg = 38
lactatemMolar = 1
hematocrit = 13.75

# bigState addends
bicarbMmoles = 8  # Set to balance-out the 5% CO2 level from the 95-5 respirator gas
gasFlowLPM = .8
gasRichness = .8
hours = 0

# Action and state dimensions
possibleActions = 7  # Number of action dimensions
possibleStates = 6   # Number of state dimensions

# Initialize the bigState with all the system states
initialBigState = [temperatureCelsius, pressuremmHg, perfusionFlowmLpm, perfusionFlowIndex, 
                   pH, pO2mmHg, pvO2, svO2, pvCO2mmHg, glucosemMolar, insulinmUnits, 
                   lactatemMolar, hematocrit, bicarbMmoles, gasFlowLPM, gasRichness, hours]

# Reset values
resetReward = 33.75  # Initial reward value

# Define the Transition tuple for experience replay
Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward', 'done'))
criticalDepletion = [10, 20, 10, .03, 6.9, 70, 0, 0, 0, 2, 1, 0, 1]
depletion = [19, 40, 20, .07, 7.5, 100, 10, .3, 20, 3, 15, 0, 10] 
excess = [38, 100, 150, 1, 7.5, 600, 500, .7, 50, 9, 45, 15, 100]
criticalExcess = [41, 120, 200, 2, 7.6, 700, 760, 1, 60, 33, 80, 30, 100]

class ReplayMemory:
    """Experience replay buffer to store and sample transitions"""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
        
    def sample(self, batch_size):
        """Sample a batch of transitions"""
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# class DQN(nn.Module):
#     """Deep Q-Network architecture"""
#     def __init__(self, state_size, action_size, hidden_size=128):
#         super(DQN, self).__init__()
#         self.fc1 = nn.Linear(state_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.fc3 = nn.Linear(hidden_size, action_size)
        
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)
    
class DQN(nn.Module):
    """Enhanced Deep Q-Network architecture"""
    def __init__(self, state_size, action_size, hidden_size=256):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)  
        
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        return self.fc4(x)

class PumpScape(Env):
    """Environment for perfusion system simulation"""
    def __init__(self):
        super(PumpScape, self).__init__()
        
        # Define continuous state space (using the 6 key parameters)
        self.observation_space = spaces.Box(
            low=np.array([20, 0.01, 6.9, 0, 2, 1, 0]), 
            high=np.array([40, 2.0, 7.6, 500, 33, 80, 24]),
            dtype=np.float32
        )
        
        # Define discrete action space (3^7 possible actions)
        self.action_space = spaces.Discrete(3**possibleActions)
        
        # Initial state
        self.state = None
        self.bigState = None
        
    def decode_state(self, bigState):
        """Extract the 6 key parameters from bigState"""
        # Map from 17-element bigState to 6-element state
        # [temperature, PFI, pH, pvO2, glucose, insulin]
        return np.array([
            bigState[0],      # temperature
            bigState[3],      # PFI
            bigState[4],      # pH
            bigState[6],      # pvO2
            bigState[9],      # glucose
            bigState[10]      # insulin      
        ], dtype=np.float32)
    
    def decode_action(self, action_idx):
        """Convert action index to action components"""
        action_vector = []
        temp = action_idx
        
        for _ in range(possibleActions):
            component = temp % 3
            if component == 2:
                component = -1  # Convert to -1, 0, 1 range
            action_vector.append(component)
            temp = temp // 3
            
        return action_vector
        
    def step(self, action_idx,train=True):
        # Decode the action index to action components
        action_vector = self.decode_action(action_idx)
        
        # Prepare the action combo for SingleStep
        action_combo = [action_idx, self.bigState]
        
        # Call the simulator to get the next state and reward
        answer = SingleStep(action_combo)
        
        # Unpack the results
        self.bigState = answer[0]  # Updated bigState
        score_vector = answer[1]   # Score vector
        simulator_reward = answer[2]  # Simulator reward
        # reward = answer[2]         # Reward value
        
        # Extract the 6-element state from bigState
        self.state = self.decode_state(self.bigState)

        hours_survived = self.bigState[16]
        # time_bonus = hours_survived * 10  # Reward for surviving longer
        # reward += time_bonus
        
        # Check if episode is done (24 hours reached or critical values)
        done = False
        if hours_survived >= 24:  # Hours exceed 24
            done = True
        
        # Check if any critical values were reached
        counts = 0
        for score in score_vector:
            if abs(score) >= 2:  # Critical values have score -2 or 2
                done = True
                break
                # counts += 1
            # if counts >= 4 and not train:
            #     done = True
            #     break
        # Calculate reward only at the end of the episode
        if done:
            # Episodic reward based on how long the organ survived
            # Scale it to make longer survival much more rewarding
            reward = hours_survived * hours_survived  # Quadratic scaling to emphasize longer survival
            
            # If it ended due to critical values but survived a decent time, still give some reward
            if hours_survived > 12 and hours_survived < 24:
                # Still reward for lasting more than half the time
                reward = hours_survived * 5  # Linear scaling for partial success
            elif hours_survived < 12:
                # Small reward or penalty for early failure
                reward = hours_survived - 10  # Penalty for early failure
        else:
            # No immediate reward during episode
            reward = 0
                
        return self.state, reward, done, {}
        
    def reset(self):
        # Reset to initial conditions with some randomness
        self.bigState = initialBigState.copy()
        
        # Add some random variation to initial vascular resistance
        vascularResistance = random.uniform(0.8, 1.2)
        perfusionFlowmLpm = pressuremmHg / vascularResistance
        perfusionFlowIndex = 100 / (vascularResistance * 300)
        
        self.bigState[2] = perfusionFlowmLpm
        self.bigState[3] = perfusionFlowIndex
        self.bigState[16] = 0  # Reset hours to 0
        
        # Extract the state
        self.state = self.decode_state(self.bigState)
        
        return self.state

class DQNAgent:
    """Deep Q-Network agent"""
    def __init__(self, state_size, action_size, 
                 lr=1e-4, gamma=0.99, epsilon_start=1.0, 
                 epsilon_end=0.01, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64,
                 update_every=4):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # Discount factor
        
        # Epsilon parameters for exploration
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Training parameters
        self.batch_size = batch_size
        self.update_every = update_every
        self.losses = []
        
        # Neural networks: policy network and target network
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only used for inference
        
        # Replay memory
        self.memory = ReplayMemory(buffer_size)
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Step counter
        self.t_step = 0
    
    def choose_action(self, state):
        """Choose an action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        # Convert state to tensor
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        # Get action values from policy network
        self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net(state)
        self.policy_net.train()
        
        # Return the action with highest value
        return action_values.argmax().item()
    
    def learn(self):
        """Update the network parameters using a batch of experiences"""
        # If not enough samples in memory, return
        if len(self.memory) < self.batch_size:
            return
        
        # Sample a batch of transitions
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Create tensors for each element
        state_batch = torch.FloatTensor(np.array(batch.state)).to(device)
        action_batch = torch.LongTensor(np.array(batch.action)).unsqueeze(1).to(device)
        reward_batch = torch.FloatTensor(np.array(batch.reward)).unsqueeze(1).to(device)
        
        # Handle terminal states and next states
        non_terminal_mask = torch.BoolTensor(
            tuple(map(lambda s: s is not None, batch.next_state))).to(device)
        
        next_states = [s for s in batch.next_state if s is not None]
        next_state_batch = torch.FloatTensor(np.array(next_states)).to(device)
        
        # Compute current Q values
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute next state values
        next_state_values = torch.zeros(self.batch_size, 1).to(device)
        with torch.no_grad():
            next_state_values[non_terminal_mask] = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
        
        # Compute expected Q values
        expected_state_action_values = reward_batch + (self.gamma * next_state_values)
        
        # Compute loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        self.losses.append(loss.item())
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
            
        self.optimizer.step()
    
    def update_target_network(self):
        """Update the target network with the policy network weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def step(self, state, action, reward, next_state, done):
        """Store experience in replay memory and learn if needed"""
        # Save experience in replay memory
        self.memory.push(state, action, next_state if not done else None, reward, done)
        
        # Increment step counter
        self.t_step += 1
        
        # Learn every update_every steps
        if self.t_step % self.update_every == 0:
            if len(self.memory) > self.batch_size:
                self.learn()
                
        # Update target network periodically
        if self.t_step % (self.update_every * 10) == 0:
            self.update_target_network()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

def train_agent(env, agent, num_episodes=15000, max_steps=24, 
                print_every=100, early_stopping_threshold=200):
    """Train the DQN agent"""
    rewards = []
    avg_rewards = []
    best_avg_reward = -float('inf')
    patience_counter = 0
    epsilon_history = []
    max_steps = 24  # Max steps for evaluation
    episode_durations = []

    all_actions_idx = [[] for _ in range(max_steps)]
    all_actions_components = [[] for _ in range(max_steps)]
    
    for episode in range(1, num_episodes+1):
        state = env.reset()
        total_reward = 0
        step_count = 0

        for t in range(max_steps):
            # Choose and take action
            action = agent.choose_action(state)
            action_idx = action
            action_components = env.decode_action(action_idx)

            all_actions_idx[step_count].append(action_idx)
            all_actions_components[step_count].append(action_components)

            next_state, reward, done, _ = env.step(action)
            
            # Save experience and learn
            agent.step(state, action, reward, next_state, done)
            
            # Update state and rewards
            state = next_state
            total_reward += reward
            step_count += 1
            
            if done:
                break
                
        rewards.append(total_reward)
        episode_durations.append(step_count)
        
        # Calculate average reward over last 100 episodes
        avg_reward = np.mean(rewards[-100:])
        avg_rewards.append(avg_reward)

        epsilon_history.append(agent.epsilon)
        
        # Print progress
        if episode % print_every == 0:
            print(f"Episode {episode}/{num_episodes} | Avg Reward: {avg_reward:.2f} | Epsilon: {agent.epsilon:.4f}")
    
    action_component_means = []
    
    for hour in range(max_steps):
        if all_actions_components[hour]:
            # Convert list of action component lists to numpy array
            hour_actions = np.array(all_actions_components[hour])
            # Calculate mean for each action component
            action_component_means.append(np.mean(hour_actions, axis=0))
        else:
            action_component_means.append(np.full(possibleActions, np.nan))
    
    action_component_means = np.array(action_component_means)

    action_names = ["Temperature", "Gas Flow", "Gas Richness", 
                   "Glucose", "Insulin", "Bicarb", "Vasodilator"]

    plot_action_evolution(action_component_means, action_names, max_steps)

    # Save final model
    torch.save(agent.policy_net.state_dict(), os.path.join(output_dir, 'final_dqn_model.pth'))
    
    # Plot training progress
    plt.figure(figsize=(12, 8))
    plt.plot(rewards, alpha=0.3)
    plt.plot(avg_rewards, color='red')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Progress')
    plt.savefig(os.path.join(output_dir, 'training_progress.png'))
    
    # Plot loss history
    plt.figure(figsize=(12, 8))
    plt.plot(agent.losses)
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(os.path.join(output_dir, 'training_loss.png'))

    # Additional plots for episode duration
    plt.figure(figsize=(12, 8))
    plt.plot(episode_durations, alpha=0.3)
    plt.plot(np.convolve(episode_durations, np.ones(100)/100, mode='valid'), color='red')
    plt.xlabel('Episode')
    plt.ylabel('Duration (hours)')
    plt.title('Episode Duration During Training')
    plt.savefig(os.path.join(output_dir, 'episode_duration.png'))
    
    return rewards, avg_rewards

def evaluate_agent(env, agent, num_episodes=100, render=False):
    """Evaluate the trained agent"""
    rewards = []
    steps = []
    pfi_values = []
    action_counts = np.zeros(3**possibleActions)
    final_bigStates = []
    max_steps = 24  # Max steps for evaluation


    all_state_values = [[] for _ in range(max_steps)]

    all_actions_idx = [[] for _ in range(max_steps)]
    all_actions_components = [[] for _ in range(max_steps)]


    # For tracking a single episode in detail
    single_episode_states = []
    single_episode_actions = []
    episode_to_track = 0  # Track the first episode by default
    
    # Load best model if available
    best_model_path = os.path.join(output_dir, 'best_dqn_model.pth')
    if os.path.exists(best_model_path):
        agent.policy_net.load_state_dict(torch.load(best_model_path))
        print("Loaded best model for evaluation")
    
    # Set to evaluation mode
    agent.policy_net.eval()
    agent.epsilon = 0.0  # No exploration during evaluation
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        step_count = 0

        # Lists to store this episode's data
        episode_states = [env.bigState.copy()]  # Initial state
        episode_actions = []
        
        done = False
        while not done and step_count < 24:  # Max 24 hours
            action = agent.choose_action(state)
            action_idx = action
            action_components = env.decode_action(action_idx)

            episode_actions.append(action_components)

            all_actions_idx[step_count].append(action_idx)
            all_actions_components[step_count].append(action_components)

            action_counts[action] += 1
            next_state, reward, done, _ = env.step(action, train=False)

            episode_states.append(env.bigState.copy())

            all_state_values[step_count].append(env.bigState.copy())
            
            pfi_values.append(env.bigState[3])

            state = next_state
            total_reward += reward
            step_count += 1
        

        # Save a specific episode for detailed visualization
        # You could choose the episode based on different criteria:
        # - The longest episode
        # - An episode that reached a specific threshold
        # - A random episode
        # Here we'll choose the episode with the highest reward
        if episode == episode_to_track or total_reward > max(rewards, default=0):
            single_episode_states = episode_states
            single_episode_actions = episode_actions
            episode_to_track = episode  # Track which episode we're visualizing
            
        rewards.append(total_reward)
        steps.append(step_count)
        final_bigStates.append(env.bigState.copy())
    
    mean_state_values = []
    std_state_values = []
    
    for hour in range(max_steps):
        if all_state_values[hour]:  # Check if we have data for this hour
            hour_data = np.array(all_state_values[hour])
            mean_state_values.append(np.mean(hour_data, axis=0))
            std_state_values.append(np.std(hour_data, axis=0))
        else:
            # If no episodes reached this hour, use NaN values
            mean_state_values.append(np.full(len(initialBigState), np.nan))
            std_state_values.append(np.full(len(initialBigState), np.nan))
    
    mean_state_values = np.array(mean_state_values)
    std_state_values = np.array(std_state_values)


    action_component_means = []
    
    for hour in range(max_steps):
        if all_actions_components[hour]:
            # Convert list of action component lists to numpy array
            hour_actions = np.array(all_actions_components[hour])
            # Calculate mean for each action component
            action_component_means.append(np.mean(hour_actions, axis=0))
        else:
            action_component_means.append(np.full(possibleActions, np.nan))
    
    action_component_means = np.array(action_component_means)


        
    avg_reward = np.mean(rewards)
    avg_steps = np.mean(steps)
    
    print(f"Evaluation Results:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Steps: {avg_steps:.2f}")
    
    # Plot action distribution
    action_probs = action_counts / action_counts.sum()
    top_actions = np.argsort(-action_probs)[:10]  # Top 10 most frequent actions


    final_bigStates_arr = np.array(final_bigStates)
    mean_values = np.mean(final_bigStates_arr, axis=0)
    std_values = np.std(final_bigStates_arr, axis=0)
    
    # Parameter names for display (in the same order as defined in initialBigState)
    param_names = ["Temperature (C)", "Pressure (mmHg)", "Perfusion Flow (mLpm)", 
                   "Perfusion Flow Index", "pH", "pO2 (mmHg)", "pvO2", "svO2", 
                   "pvCO2 (mmHg)", "Glucose (mM)", "Insulin (mU)", "Lactate (mM)", 
                   "Hematocrit", "Bicarb (mmoles)", "Gas Flow (LPM)", "Gas Richness", "Hours"]

    action_names = ["Temperature", "Gas Flow", "Gas Richness", 
                   "Glucose", "Insulin", "Bicarb", "Vasodilator"]

    plot_state_evolution(mean_state_values, std_state_values, param_names, max_steps)
    plot_action_evolution(action_component_means, action_names, max_steps)

    plot_single_episode_state_evolution(single_episode_states, param_names)
    plot_single_episode_action_evolution(single_episode_actions, action_names)
    
    plt.figure(figsize=(14, 8))
    bars = plt.bar(range(len(mean_values)), mean_values, yerr=std_values, capsize=5)
    plt.xticks(range(len(mean_values)), param_names, rotation=45, ha="right")
    plt.ylabel("Average Final Value")
    plt.title("Average Final System State Parameters After Evaluation (Q-net)")
    
    # Annotate each bar with its mean value
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_state_parameters_qnet.png'))
    plt.show()
    
    return rewards, steps, action_counts


def plot_state_evolution(mean_values, std_values, param_names, max_steps):
    """Plot the evolution of each state parameter over time"""
    # Create a figure with multiple subplots
    fig, axs = plt.subplots(6, 3, figsize=(18, 24))
    axs = axs.flatten()
    
    hours = range(1, max_steps + 1)
    
    # Plot each parameter
    for i, param_name in enumerate(param_names):
        if i < len(axs):  # Ensure we don't exceed available subplots
            ax = axs[i]
            
            # Plot mean with error bands
            ax.plot(hours, mean_values[:, i], 'b-', label='Mean')
            ax.fill_between(
                hours, 
                mean_values[:, i] - std_values[:, i],
                mean_values[:, i] + std_values[:, i],
                alpha=0.2, color='b', label='Std Dev'
            )
            
            ax.set_title(param_name)
            ax.set_xlabel('Hour')
            ax.set_ylabel('Value')
            ax.grid(True, linestyle='--', alpha=0.7)
    
    # Hide any unused subplots
    for i in range(len(param_names), len(axs)):
        axs[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'state_evolution_over_time.png'))
    plt.show()


def plot_action_evolution(action_means, action_names, max_steps):
    """Plot the evolution of actions taken over time"""
    # Create a figure with subplots
    fig, axs = plt.subplots(3, 3, figsize=(18, 12))
    axs = axs.flatten()
    
    hours = range(1, max_steps + 1)
    
    # Define color mapping for action values
    colors = {-1: 'red', 0: 'gray', 1: 'green'}
    action_labels = {-1: 'Decrease', 0: 'No Change', 1: 'Increase'}
    
    # Plot each action component
    for i, action_name in enumerate(action_names):
        if i < len(axs):
            ax = axs[i]
            
            # Calculate frequency of each action value (-1, 0, 1) at each hour
            action_data = action_means[:, i]
            
            # Plot mean action value
            ax.plot(hours, action_data, 'b-o', label='Mean Action Value')
            
            # Add a horizontal line at y=0
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            
            # Add colored background based on action value
            for h in range(len(hours)):
                if not np.isnan(action_data[h]):
                    # Determine color based on if value is closer to -1, 0, or 1
                    if action_data[h] < -0.33:
                        color = colors[-1]
                    elif action_data[h] < 0.33:
                        color = colors[0]
                    else:
                        color = colors[1]
                    
                    # Add colored background rectangle
                    ax.axvspan(hours[h]-0.5, hours[h]+0.5, alpha=0.2, color=color)
            
            ax.set_title(f'Action: {action_name}')
            ax.set_xlabel('Hour')
            ax.set_ylabel('Mean Action Value (-1 to 1)')
            ax.set_ylim(-1.1, 1.1)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add a custom legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=colors[-1], alpha=0.2, label=action_labels[-1]),
                Patch(facecolor=colors[0], alpha=0.2, label=action_labels[0]),
                Patch(facecolor=colors[1], alpha=0.2, label=action_labels[1])
            ]
            ax.legend(handles=legend_elements)
    
    # Hide any unused subplots
    for i in range(len(action_names), len(axs)):
        axs[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'action_evolution_over_time.png'))
    plt.show()

    # Additional visualization: Action heatmap
    plt.figure(figsize=(12, 8))
    plt.imshow(action_means.T, aspect='auto', cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar(label='Action Value (-1 to 1)')
    plt.yticks(range(len(action_names)), action_names)
    plt.xticks(range(0, max_steps, 2), range(1, max_steps+1, 2))
    plt.xlabel('Hour')
    plt.ylabel('Action Component')
    plt.title('Action Heatmap Over Time')
    plt.grid(False)
    plt.savefig(os.path.join(output_dir, 'action_heatmap.png'))
    plt.show()

def plot_single_episode_state_evolution(states, param_names):
    """Plot the evolution of state parameters for a single episode"""
    # Convert states list to numpy array
    states_array = np.array(states)
    
    # Create a figure with multiple subplots
    fig, axs = plt.subplots(6, 3, figsize=(18, 24))
    axs = axs.flatten()
    
    hours = range(len(states))
    
    # Plot each parameter
    for i, param_name in enumerate(param_names):
        if i < len(axs):  # Ensure we don't exceed available subplots
            ax = axs[i]
            
            # Plot parameter value over time
            ax.plot(hours, states_array[:, i], 'b-o', label='Value')
            
            # Add horizontal lines for critical thresholds if applicable
            if i < 13:  # Only the first 13 parameters have thresholds
                # Critical depletion threshold
                ax.axhline(y=criticalDepletion[i], color='r', linestyle='--', alpha=0.5, label='Critical Low')
                
                # Depletion threshold
                ax.axhline(y=depletion[i], color='orange', linestyle='--', alpha=0.5, label='Low')
                
                # Excess threshold
                ax.axhline(y=excess[i], color='orange', linestyle='--', alpha=0.5, label='High')
                
                # Critical excess threshold
                ax.axhline(y=criticalExcess[i], color='r', linestyle='--', alpha=0.5, label='Critical High')
            
            ax.set_title(param_name)
            ax.set_xlabel('Hour')
            ax.set_ylabel('Value')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add legend for the first plot
            if i == 0:
                ax.legend()
    
    # Hide any unused subplots
    for i in range(len(param_names), len(axs)):
        axs[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'single_episode_state_evolution.png'))
    plt.show()


def plot_single_episode_action_evolution(actions, action_names):
    """Plot the evolution of actions for a single episode"""
    # Convert actions list to numpy array
    actions_array = np.array(actions)
    
    # Create a figure with subplots
    fig, axs = plt.subplots(3, 3, figsize=(18, 12))
    axs = axs.flatten()
    
    hours = range(1, len(actions) + 1)
    
    # Define color mapping for action values
    colors = {-1: 'red', 0: 'gray', 1: 'green'}
    action_labels = {-1: 'Decrease', 0: 'No Change', 1: 'Increase'}
    
    # Plot each action component
    for i, action_name in enumerate(action_names):
        if i < len(axs):
            ax = axs[i]
            
            # Plot action values
            ax.plot(hours, actions_array[:, i], 'b-o', label='Action Value')
            
            # Add a horizontal line at y=0
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            
            # Add colored background based on action value
            for h in range(len(hours)):
                action_val = actions_array[h, i]
                color = colors[action_val]  # Direct mapping since actions are -1, 0, or 1
                
                # Add colored background rectangle
                ax.axvspan(hours[h]-0.5, hours[h]+0.5, alpha=0.2, color=color)
            
            ax.set_title(f'Action: {action_name}')
            ax.set_xlabel('Hour')
            ax.set_ylabel('Action Value (-1 to 1)')
            ax.set_ylim(-1.1, 1.1)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add a custom legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=colors[-1], alpha=0.2, label=action_labels[-1]),
                Patch(facecolor=colors[0], alpha=0.2, label=action_labels[0]),
                Patch(facecolor=colors[1], alpha=0.2, label=action_labels[1])
            ]
            ax.legend(handles=legend_elements)
    
    # Hide any unused subplots
    for i in range(len(action_names), len(axs)):
        axs[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'single_episode_action_evolution.png'))
    plt.show()
    
    # Additional visualization: Action heatmap
    plt.figure(figsize=(12, 8))
    plt.imshow(actions_array.T, aspect='auto', cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar(label='Action Value (-1 to 1)')
    plt.yticks(range(len(action_names)), action_names)
    plt.xticks(range(0, len(actions), 2), range(1, len(actions)+1, 2))
    plt.xlabel('Hour')
    plt.ylabel('Action Component')
    plt.title('Single Episode Action Heatmap')
    plt.grid(False)
    plt.savefig(os.path.join(output_dir, 'single_episode_action_heatmap.png'))
    plt.show()


if __name__ == "__main__":
    # Create environment
    env = PumpScape()
    
    # Get state and action dimensions
    state_size = 6  # The 6 key parameters we're tracking
    action_size = 3**possibleActions  # 3^7 possible actions
    
    # Create agent
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        lr=5e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.0001,
        epsilon_decay=0.995,
        buffer_size=10000000,
        batch_size=64,
        update_every=4
    )
    
    # Train agent
    print("Starting training...")
    train_rewards, avg_rewards = train_agent(
        env=env,
        agent=agent,
        num_episodes=8000,
        max_steps=24,
        print_every=100
    )
    
    # Evaluate agent
    print("Starting evaluation...")
    eval_rewards, eval_steps, action_counts = evaluate_agent(
        env=env,
        agent=agent,
        num_episodes=300
    )
    
    print("Training and evaluation complete!")