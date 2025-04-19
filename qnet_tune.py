import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import random
import os
import time
import pandas as pd
from sklearn.model_selection import ParameterGrid

# Configure output directory
output_dir = os.path.expanduser("~/SimulatorOutput/")
os.makedirs(output_dir, exist_ok=True)

# Import the environment and simulator
from FunctionalStepSimulator import SingleStep
from gym import Env, spaces

# Keep the environment definition from the previous code
# ...

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

class DQN(nn.Module):
    """Deep Q-Network architecture"""
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class PumpScape(Env):
    """Environment for perfusion system simulation"""
    def __init__(self):
        super(PumpScape, self).__init__()
        
        # Define continuous state space (using the 6 key parameters)
        self.observation_space = spaces.Box(
            low=np.array([20, 0.01, 6.9, 0, 2, 1]), 
            high=np.array([40, 2.0, 7.6, 500, 33, 80]),
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
        reward = answer[2]         # Reward value
        
        # Extract the 6-element state from bigState
        self.state = self.decode_state(self.bigState)
        
        # Check if episode is done (24 hours reached or critical values)
        done = False
        if self.bigState[16] >= 24:  # Hours exceed 24
            done = True
        
        # Check if any critical values were reached
        counts = 0
        for score in score_vector:
            if abs(score) >= 2 and not train:  # Critical values have score -2 or 2
                done = True
                break
                # counts += 1
            # if counts >= 3 and not train:
            #     done = True
            #     break
                
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

# Modified DQN class with additional features
class EnhancedDQN(nn.Module):
    """Enhanced Deep Q-Network architecture with configurable layers and dropout"""
    def __init__(self, state_size, action_size, hidden_sizes=[256, 256], dropout_rate=0.2):
        super(EnhancedDQN, self).__init__()
        
        # Create a list to hold all layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(state_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_sizes)-1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], action_size))
        
        # Combine all layers into a sequential model
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

# Modified DQN Agent with additional hyperparameters
class TunableDQNAgent:
    """DQN agent with tunable hyperparameters"""
    def __init__(self, 
                 state_size, 
                 action_size,
                 hidden_sizes=[128, 128],
                 dropout_rate=0.0,
                 lr=1e-4, 
                 gamma=0.99, 
                 epsilon_start=1.0,
                 epsilon_end=0.01, 
                 epsilon_decay=0.995,
                 buffer_size=100000, 
                 batch_size=64,
                 update_every=4,
                 target_update_freq=10,
                 reward_shaping=False,
                 double_dqn=False,
                 dueling_dqn=False,
                 prioritized_replay=False,
                 reward_clip=None):
        
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.update_every = update_every
        self.target_update_freq = target_update_freq
        self.reward_shaping = reward_shaping
        self.double_dqn = double_dqn
        self.reward_clip = reward_clip
        
        # Epsilon parameters
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Initialize networks
        self.policy_net = EnhancedDQN(state_size, action_size, hidden_sizes, dropout_rate).to(device)
        self.target_net = EnhancedDQN(state_size, action_size, hidden_sizes, dropout_rate).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Initialize replay memory
        self.memory = ReplayMemory(buffer_size)
        
        # Training metrics
        self.losses = []
        self.t_step = 0
        self.episode_steps = []
        self.episode_rewards = []
        
    def choose_action(self, state):
        """Choose action using epsilon-greedy policy with decay"""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net(state)
        self.policy_net.train()
        
        return action_values.argmax().item()
    
    def step(self, state, action, reward, next_state, done):
        """Process a step and learn if needed"""
        
        # Apply reward shaping if enabled
        if self.reward_shaping:
            # Add small positive reward for surviving another step
            shaped_reward = reward + 0.1
            
            # Extra penalty for early termination
            if done and next_state is not None:  # Not at max steps
                shaped_reward -= 50
        else:
            shaped_reward = reward
            
        # Apply reward clipping if enabled
        if self.reward_clip is not None:
            shaped_reward = max(min(shaped_reward, self.reward_clip), -self.reward_clip)
        
        # Store experience in replay memory
        self.memory.push(state, action, next_state if not done else None, shaped_reward, done)
        
        # Learn every update_every time steps
        self.t_step += 1
        if self.t_step % self.update_every == 0:
            if len(self.memory) > self.batch_size:
                self.learn()
                
        # Update target network periodically
        if self.t_step % (self.update_every * self.target_update_freq) == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def learn(self):
        """Update policy network parameters using batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
            
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Create tensors for each element
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), 
                                      device=device, dtype=torch.bool)
        
        non_final_next_states = torch.FloatTensor([s for s in batch.next_state if s is not None]).to(device)
        state_batch = torch.FloatTensor(batch.state).to(device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(device)
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(device)
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute V(s_{t+1}) for all next states
        next_state_values = torch.zeros(self.batch_size, 1, device=device)
        
        if self.double_dqn:
            # Double DQN: Use policy network to select actions, target network to evaluate them
            with torch.no_grad():
                # Get actions from policy network
                next_actions = self.policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
                # Evaluate with target network
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_actions)
        else:
            # Standard DQN
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].unsqueeze(1)
        
        # Compute the expected Q values
        expected_state_action_values = reward_batch + (self.gamma * next_state_values)
        
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        self.losses.append(loss.item())
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update epsilon with decay
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'losses': self.losses,
            'episode_steps': self.episode_steps,
            'episode_rewards': self.episode_rewards
        }
        torch.save(checkpoint, filename)
        
    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        checkpoint = torch.load(filename)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.losses = checkpoint['losses']
        self.episode_steps = checkpoint['episode_steps']
        self.episode_rewards = checkpoint['episode_rewards']

def hyperparameter_grid_search():
    """Set up and perform grid search for optimal hyperparameters"""
    
    # Define parameter grid for search
    param_grid = {
        'hidden_sizes': [[128, 128], [256, 256], [128, 128, 128]],
        'dropout_rate': [0.0, 0.1, 0.2],
        'lr': [1e-4, 5e-4, 1e-3],
        'gamma': [0.95, 0.99, 0.995],
        'epsilon_decay': [0.995, 0.997, 0.999],  # Slower decay for more exploration
        'batch_size': [32, 64, 128],
        'update_every': [1, 4, 8],
        'target_update_freq': [5, 10, 20],
        'reward_shaping': [True, False],
        'double_dqn': [True, False]
    }
    
    # For efficiency, we'll use a smaller subset for demonstration
    # In a real scenario, you could run a full grid search or use random search
    reduced_param_grid = {
        'hidden_sizes': [[128, 128], [256, 256]],
        'lr': [1e-4, 5e-4],
        'gamma': [0.99, 0.995],  # Higher discount factors for longer-term planning
        'epsilon_decay': [0.997, 0.999],  # Slower decay for more exploration
        'reward_shaping': [True],  # Enable reward shaping to encourage longer episodes
        'double_dqn': [True]  # Use Double DQN to prevent overestimation
    }
    
    # Generate all parameter combinations 
    grid = list(ParameterGrid(reduced_param_grid))
    print(f"Total configurations to test: {len(grid)}")
    
    # Initialize environment
    env = PumpScape()
    
    # Parameters for training during grid search
    search_episodes = 3000  # Reduced number for faster search
    max_steps = 24
    
    # Results tracking
    results = []
    
    # Run each configuration
    for idx, params in enumerate(grid):
        print(f"Testing configuration {idx+1}/{len(grid)}: {params}")
        
        # Create agent with current hyperparameters
        agent = TunableDQNAgent(
            state_size=6,
            action_size=3**7,
            hidden_sizes=params['hidden_sizes'],
            lr=params['lr'],
            gamma=params['gamma'],
            epsilon_decay=params['epsilon_decay'],
            reward_shaping=params['reward_shaping'],
            double_dqn=params['double_dqn']
        )
        
        # Train agent
        train_rewards, train_steps = train_agent(
            env=env,
            agent=agent,
            num_episodes=search_episodes,
            max_steps=max_steps,
            print_every=100,
            early_stopping_patience=100,  # Disable early stopping for grid search
            save_checkpoints=False
        )
        
        # Evaluate the trained agent
        eval_rewards, eval_steps = evaluate_agent(
            env=env,
            agent=agent,
            num_episodes=50
        )
        
        # Record results
        results.append({
            'params': params,
            'avg_train_reward': np.mean(train_rewards[-100:]),
            'avg_train_steps': np.mean(train_steps[-100:]),
            'avg_eval_reward': np.mean(eval_rewards),
            'avg_eval_steps': np.mean(eval_steps)
        })
        
        # Save results after each configuration in case of crashes
        save_results(results, os.path.join(output_dir, 'hyperparameter_search_results.csv'))
    
    # Find best configuration
    best_config = max(results, key=lambda x: x['avg_eval_steps'])
    print(f"Best configuration: {best_config['params']}")
    print(f"Average evaluation steps: {best_config['avg_eval_steps']}")
    
    return best_config

def train_agent(env, agent, num_episodes=2000, max_steps=24, 
                print_every=100, early_stopping_patience=200,
                save_checkpoints=True):
    """Train the DQN agent with improved monitoring"""
    rewards = []
    steps = []
    best_avg_steps = 0
    patience_counter = 0
    
    for episode in range(1, num_episodes+1):
        state = env.reset()
        total_reward = 0
        episode_steps = 0
        
        for t in range(max_steps):
            # Choose action and take step
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            
            # Store experience and learn
            agent.step(state, action, reward, next_state, done)
            
            # Update tracking variables
            state = next_state
            total_reward += reward
            episode_steps += 1
            
            if done:
                break
        
        # Record episode results
        rewards.append(total_reward)
        steps.append(episode_steps)
        agent.episode_steps.append(episode_steps)
        agent.episode_rewards.append(total_reward)
        
        # Calculate moving averages
        avg_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
        avg_steps = np.mean(steps[-100:]) if len(steps) >= 100 else np.mean(steps)
        
        # Print progress
        if episode % print_every == 0:
            print(f"Episode {episode}/{num_episodes} | Avg Reward: {avg_reward:.2f} | Avg Steps: {avg_steps:.2f} | Epsilon: {agent.epsilon:.4f}")
        
        # Check for improvement in average steps
        # if avg_steps > best_avg_steps:
        #     best_avg_steps = avg_steps
        #     patience_counter = 0
            
        #     # Save best model
        #     if save_checkpoints:
        #         torch.save(agent.policy_net.state_dict(), os.path.join(output_dir, 'best_dqn_model.pth'))
        # else:
        #     patience_counter += 1
        
        # Early stopping check
        # if patience_counter >= early_stopping_patience and early_stopping_patience > 0:
        #     print(f"Early stopping at episode {episode} due to no improvement in average steps")
        #     break
        
        # Periodic checkpoint saving
        if save_checkpoints and episode % 500 == 0:
            agent.save_checkpoint(os.path.join(output_dir, f'dqn_checkpoint_{episode}.pth'))
    
    # Create training plots
    plot_training_results(agent, rewards, steps)
    
    return rewards, steps

def evaluate_agent(env, agent, num_episodes=100, render=False):
    """Evaluate the trained agent with focus on episode length"""
    rewards = []
    steps = []
    
    # Set to evaluation mode
    agent.policy_net.eval()
    agent.epsilon = 0.0  # No exploration during evaluation
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        episode_steps = 0
        
        done = False
        while not done and episode_steps < 24:  # Max 24 hours
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action, train=False)
            
            state = next_state
            total_reward += reward
            episode_steps += 1
            
        rewards.append(total_reward)
        steps.append(episode_steps)
    
    # Calculate statistics
    avg_reward = np.mean(rewards)
    avg_steps = np.mean(steps)
    median_steps = np.median(steps)
    max_steps = np.max(steps)
    min_steps = np.min(steps)
    
    print(f"Evaluation Results:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Median Steps: {median_steps}")
    print(f"Step Range: {min_steps} - {max_steps}")
    
    # Plot step distribution
    plt.figure(figsize=(10, 6))
    plt.hist(steps, bins=24, alpha=0.7)
    plt.axvline(avg_steps, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {avg_steps:.2f}')
    plt.axvline(median_steps, color='g', linestyle='dashed', linewidth=2, label=f'Median: {median_steps}')
    plt.xlabel('Steps')
    plt.ylabel('Count')
    plt.title('Distribution of Episode Lengths During Evaluation')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'episode_length_distribution.png'))
    
    return rewards, steps

def plot_training_results(agent, rewards, steps):
    """Create comprehensive training progress plots"""
    # Plot rewards
    plt.figure(figsize=(12, 8))
    plt.plot(rewards, alpha=0.3, label='Individual Episodes')
    plt.plot(np.convolve(rewards, np.ones(100)/100, mode='valid'), color='red', label='100-Episode Average')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'training_rewards.png'))
    
    # Plot steps
    plt.figure(figsize=(12, 8))
    plt.plot(steps, alpha=0.3, label='Individual Episodes')
    plt.plot(np.convolve(steps, np.ones(100)/100, mode='valid'), color='red', label='100-Episode Average')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Episode Lengths During Training')
    plt.axhline(y=24, color='green', linestyle='-', label='Maximum Steps')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'training_steps.png'))
    
    # Plot loss
    if agent.losses:
        plt.figure(figsize=(12, 8))
        plt.plot(agent.losses)
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig(os.path.join(output_dir, 'training_loss.png'))
    
    # Plot epsilon decay
    epsilons = [agent.epsilon_start * (agent.epsilon_decay ** i) for i in range(len(rewards))]
    epsilons = [max(e, agent.epsilon_end) for e in epsilons]
    
    plt.figure(figsize=(12, 6))
    plt.plot(epsilons)
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Decay')
    plt.savefig(os.path.join(output_dir, 'epsilon_decay.png'))

def save_results(results, filename):
    """Save grid search results to CSV"""
    # Convert results to dataframe
    df_rows = []
    for result in results:
        row = {
            'avg_train_reward': result['avg_train_reward'],
            'avg_train_steps': result['avg_train_steps'],
            'avg_eval_reward': result['avg_eval_reward'],
            'avg_eval_steps': result['avg_eval_steps']
        }
        # Add parameters
        for key, value in result['params'].items():
            if isinstance(value, list):
                row[key] = str(value)  # Convert lists to strings for CSV
            else:
                row[key] = value
        
        df_rows.append(row)
    
    df = pd.DataFrame(df_rows)
    df.to_csv(filename, index=False)
    
def train_with_best_params(best_config, num_episodes=2000):
    """Train final model with best hyperparameters"""
    env = PumpScape()
    
    # Create agent with best parameters
    agent = TunableDQNAgent(
        state_size=6,
        action_size=3**7,
        **best_config['params']
    )
    
    # Train the agent longer with best parameters
    rewards, steps = train_agent(
        env=env,
        agent=agent,
        num_episodes=num_episodes,
        max_steps=24,
        print_every=100,
        early_stopping_patience=300
    )
    
    # Final evaluation
    eval_rewards, eval_steps = evaluate_agent(
        env=env,
        agent=agent,
        num_episodes=200
    )
    
    # Save final model
    agent.save_checkpoint(os.path.join(output_dir, 'final_model.pth'))
    
    return agent


if __name__ == "__main__":
    # Check if CUDA is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Run hyperparameter search
    print("Starting hyperparameter grid search...")
    best_config = hyperparameter_grid_search()
    
    # Train final model with best parameters
    print("Training final model with best parameters...")
    final_agent = train_with_best_params(best_config, num_episodes=4000)
    
    print("Training complete!")

    print("Evaluating the trained agent...")
    eval_rewards, eval_steps = evaluate_agent(
        env=PumpScape(),
        agent=final_agent,
        num_episodes=100
    )
    
    print(f"Final evaluation complete! Average steps: {np.mean(eval_steps):.2f}")