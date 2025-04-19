# Modified PumpScape environment with flexible critical value handling
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
    """Environment with configurable tolerance for critical values"""
    def __init__(self, training_mode=True):
        super(PumpScape, self).__init__()
        
        # Define observation and action spaces as before
        self.observation_space = spaces.Box(
            low=np.array([20, 0.01, 6.9, 0, 2, 1]), 
            high=np.array([40, 2.0, 7.6, 500, 33, 80]),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3**possibleActions)
        
        # Initial state and mode
        self.state = None
        self.bigState = None
        self.training_mode = training_mode
        
        # Critical value tracking
        self.critical_violations = {i: 0 for i in range(13)}  # Count of violations per parameter
        self.violation_duration = {i: 0 for i in range(13)}   # Duration of each violation
        self.total_violations = 0
        self.critical_events = []  # Track when and which parameters went critical
        
        # Tolerance configuration
        self.max_violations = 5      # Maximum allowed violations before termination
        self.max_duration = 3        # Maximum consecutive hours a parameter can be critical
        self.violation_window = 8    # Window of time to count violations within
        self.critical_param_weights = {
            0: 1.0,  # Temperature
            1: 1.5,  # PFI - weighted more heavily
            2: 1.0,  # Flow
            3: 1.0,  # VR
            4: 1.2,  # pH - weighted more heavily
            5: 0.8,  # pO2
            6: 1.0,  # pvO2
            7: 0.7,  # SvO2
            8: 0.7,  # pvCO2
            9: 1.0,  # Glucose
            10: 0.5, # Insulin - less critical
            11: 0.8, # Lactate 
            12: 0.5  # Hematocrit - less critical
        }
        
    def decode_state(self, bigState):
        """Extract the 6 key parameters from bigState"""
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
                component = -1
            action_vector.append(component)
            temp = temp // 3
            
        return action_vector
        
    def step(self, action_idx):
        """Step function with flexible critical value handling"""
        # Decode the action and prepare for simulation
        action_vector = self.decode_action(action_idx)
        action_combo = [action_idx, self.bigState]
        
        # Call the simulator to get the next state and reward
        answer = SingleStep(action_combo)
        
        # Unpack the results
        self.bigState = answer[0]    # Updated bigState
        score_vector = answer[1]     # Score vector
        base_reward = answer[2]      # Base reward value
        
        # Extract the 6-element state from bigState
        self.state = self.decode_state(self.bigState)
        
        # Current hour
        current_hour = self.bigState[16]
        
        # Process critical value violations
        current_violations = 0
        for i, score in enumerate(score_vector):
            if abs(score) >= 2:  # Critical value
                self.critical_violations[i] += 1
                self.violation_duration[i] += 1
                current_violations += 1
                
                # Record the critical event
                self.critical_events.append((current_hour, i, score))
                
                # Apply weighted penalty to reward
                violation_penalty = self.critical_param_weights[i] * 10
                base_reward -= violation_penalty
            else:
                # Reset duration counter if parameter is no longer critical
                self.violation_duration[i] = 0
        
        # Calculate total violations in recent window
        recent_violations = sum(1 for event in self.critical_events 
                              if current_hour - event[0] <= self.violation_window)
        
        # Calculate reward
        reward = base_reward
        
        # Add time survival bonus
        reward += current_hour * 2
        
        # Bonus for completing the full duration
        if current_hour >= 24:
            reward += 100
        
        # Determine if episode should terminate
        done = False
        termination_reason = None
        
        if self.training_mode:
            # In training mode, only terminate after 24 hours
            done = (current_hour >= 24)
        else:
            # In evaluation mode, check termination conditions
            
            # 1. Full duration reached
            if current_hour >= 24:
                done = True
                termination_reason = "completed"
                
            # 2. Too many recent violations
            elif recent_violations > self.max_violations:
                done = True
                termination_reason = f"exceeded_violations: {recent_violations}"
                
            # 3. Any parameter critical for too long
            for param, duration in self.violation_duration.items():
                if duration > self.max_duration:
                    done = True
                    termination_reason = f"extended_violation: param {param}, duration {duration}"
                    break
        
        # Additional info for analysis
        info = {
            'current_violations': current_violations,
            'recent_violations': recent_violations,
            'violation_durations': dict(self.violation_duration),
            'termination_reason': termination_reason,
            'hours': current_hour
        }
        
        return self.state, reward, done, info
    
    def set_tolerances(self, max_violations=5, max_duration=3, violation_window=8):
        """Configure how tolerant the environment is to critical values"""
        self.max_violations = max_violations
        self.max_duration = max_duration
        self.violation_window = violation_window
    
    def set_mode(self, training=True):
        """Set the environment mode to training or evaluation"""
        self.training_mode = training
        
    def reset(self):
        """Reset the environment"""
        # Reset to initial conditions with some randomness
        self.bigState = initialBigState.copy()
        
        # Add some random variation to initial vascular resistance
        vascularResistance = random.uniform(0.8, 1.2)
        perfusionFlowmLpm = pressuremmHg / vascularResistance
        perfusionFlowIndex = 100 / (vascularResistance * 300)
        
        self.bigState[2] = perfusionFlowmLpm
        self.bigState[3] = perfusionFlowIndex
        self.bigState[16] = 0  # Reset hours to 0
        
        # Reset critical value tracking
        self.critical_violations = {i: 0 for i in range(13)}
        self.violation_duration = {i: 0 for i in range(13)}
        self.total_violations = 0
        self.critical_events = []
        
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

# Function to implement curriculum learning
def curriculum_training(env, agent, num_episodes=2000, eval_every=100):
    """Train with gradually increasing difficulty"""
    # Define curriculum stages - progressively stricter tolerances
    curriculum_stages = [
        {"stage": 1, "episodes": 2000, "max_violations": 10, "max_duration": 5, "violation_window": 12},
        {"stage": 2, "episodes": 2000, "max_violations": 7, "max_duration": 4, "violation_window": 10},
        {"stage": 3, "episodes": 2000, "max_violations": 5, "max_duration": 3, "violation_window": 8},
        {"stage": 4, "episodes": 2000, "max_violations": 3, "max_duration": 2, "violation_window": 6}
    ]
    
    total_episodes = 0
    stage_results = []
    
    for stage in curriculum_stages:
        print(f"\n===== Starting Curriculum Stage {stage['stage']} =====")
        print(f"Max violations: {stage['max_violations']}, Max duration: {stage['max_duration']}, Window: {stage['violation_window']}")
        
        # Set the tolerances for this stage
        env.set_tolerances(
            max_violations=stage['max_violations'],
            max_duration=stage['max_duration'],
            violation_window=stage['violation_window']
        )
        
        # Train for this stage
        train_rewards, eval_rewards, eval_steps, critical_stats = train_and_evaluate(
            env=env,
            agent=agent,
            num_episodes=stage['episodes'],
            eval_every=eval_every,
            stage_offset=total_episodes
        )
        
        # Save stage results
        stage_results.append({
            "stage": stage['stage'],
            "tolerances": {k: v for k, v in stage.items() if k != "stage" and k != "episodes"},
            "final_eval_reward": eval_rewards[-1] if eval_rewards else None,
            "final_eval_steps": eval_steps[-1] if eval_steps else None,
            "critical_stats": critical_stats[-1] if critical_stats else None
        })
        
        total_episodes += stage['episodes']
        
        # Save checkpoint model for this stage
        torch.save(agent.policy_net.state_dict(), 
                   os.path.join(output_dir, f'stage_{stage["stage"]}_model.pth'))
    
    # Save final model
    torch.save(agent.policy_net.state_dict(), 
               os.path.join(output_dir, 'final_curriculum_model.pth'))
    
    # Save curriculum results
    import json
    with open(os.path.join(output_dir, 'curriculum_results.json'), 'w') as f:
        json.dump(stage_results, f, indent=2)
        
    return stage_results


# Enhanced training and evaluation function
def train_and_evaluate(env, agent, num_episodes=500, max_steps=24, 
                       print_every=50, eval_every=100, eval_episodes=10,
                       stage_offset=0):
    """Train with full episodes and evaluate with flexible critical handling"""
    train_rewards = []
    eval_rewards = []
    eval_steps = []
    critical_stats = []  # Track critical violation statistics
    
    best_eval_reward = -float('inf')
    
    # Set environment to training mode
    env.set_mode(training=True)
    
    for episode in range(1, num_episodes+1):
        # TRAINING
        state = env.reset()
        episode_reward = 0
        
        for t in range(max_steps):
            # Choose and take action
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            
            # Save experience and learn
            agent.step(state, action, reward, next_state, done)
            
            # Update state and rewards
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        train_rewards.append(episode_reward)
        
        # Print training progress
        if episode % print_every == 0:
            avg_reward = np.mean(train_rewards[-100:])
            print(f"Episode {episode+stage_offset}/{num_episodes+stage_offset} | "
                  f"Training Reward: {avg_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.4f}")
        
        # EVALUATION
        if episode % eval_every == 0:
            # Switch to evaluation mode
            env.set_mode(training=False)
            eval_episode_rewards = []
            eval_episode_steps = []
            eval_violation_stats = []
            
            for _ in range(eval_episodes):
                eval_state = env.reset()
                eval_episode_reward = 0
                eval_step_count = 0
                
                while True:
                    # Use greedy policy for evaluation
                    eval_action = agent.choose_action(eval_state)
                    eval_next_state, eval_reward, eval_done, eval_info = env.step(eval_action)
                    
                    eval_state = eval_next_state
                    eval_episode_reward += eval_reward
                    eval_step_count += 1
                    
                    if eval_done:
                        # Collect violation statistics
                        eval_violation_stats.append({
                            'steps': eval_step_count,
                            'current_violations': eval_info['current_violations'],
                            'recent_violations': eval_info['recent_violations'],
                            'termination_reason': eval_info['termination_reason']
                        })
                        break
                
                eval_episode_rewards.append(eval_episode_reward)
                eval_episode_steps.append(eval_step_count)
            
            # Calculate average evaluation metrics
            avg_eval_reward = np.mean(eval_episode_rewards)
            avg_eval_steps = np.mean(eval_episode_steps)
            eval_rewards.append(avg_eval_reward)
            eval_steps.append(avg_eval_steps)
            
            # Calculate completion statistics
            completions = sum(1 for stat in eval_violation_stats 
                             if stat['termination_reason'] == 'completed')
            completion_rate = completions / eval_episodes
            
            # Collect critical statistics
            critical_stats.append({
                'episode': episode + stage_offset,
                'avg_steps': avg_eval_steps,
                'completion_rate': completion_rate,
                'violation_stats': eval_violation_stats
            })
            
            # Print evaluation results
            print(f"\nEvaluation at episode {episode+stage_offset}: "
                  f"Avg Reward: {avg_eval_reward:.2f} | "
                  f"Avg Steps: {avg_eval_steps:.2f} | "
                  f"Completion Rate: {completion_rate:.2f} ({completions}/{eval_episodes})")
            
            # Print termination reason statistics
            termination_reasons = {}
            for stat in eval_violation_stats:
                reason = stat['termination_reason']
                if reason in termination_reasons:
                    termination_reasons[reason] += 1
                else:
                    termination_reasons[reason] = 1
            
            print("Termination reasons:")
            for reason, count in termination_reasons.items():
                print(f"  {reason}: {count}")
            print()
            
            # Save best model based on evaluation performance
            if avg_eval_reward > best_eval_reward:
                best_eval_reward = avg_eval_reward
                torch.save(agent.policy_net.state_dict(), 
                           os.path.join(output_dir, 'best_dqn_model.pth'))
            
            # Switch back to training mode
            env.set_mode(training=True)
    
    return train_rewards, eval_rewards, eval_steps, critical_stats


# Run the curriculum training
if __name__ == "__main__":
    # Create environment
    env = PumpScape(training_mode=True)
    
    # Create agent
    agent = DQNAgent(
        state_size=6,
        action_size=3**possibleActions,
        lr=5e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.997,
        buffer_size=100000,
        batch_size=64,
        update_every=4
    )
    
    # Run curriculum training
    stage_results = curriculum_training(
        env=env,
        agent=agent,
        num_episodes=15000,
        eval_every=100
    )
    
    # Final evaluation with the trained agent
    print("\n===== Final Evaluation =====")
    env.set_mode(training=False)
    env.set_tolerances(max_violations=3, max_duration=2, violation_window=6)  # Strict final evaluation
    
    # Load the best model
    agent.policy_net.load_state_dict(torch.load(os.path.join(output_dir, 'best_dqn_model.pth')))
    agent.epsilon = 0.0  # No exploration during final evaluation
    
    # Run 50 evaluation episodes
    _, _, _, final_stats = train_and_evaluate(
        env=env,
        agent=agent,
        num_episodes=1,  # Just one "training" episode to trigger evaluation
        eval_episodes=300,
        eval_every=1
    )
    
    print("Training and evaluation complete!")