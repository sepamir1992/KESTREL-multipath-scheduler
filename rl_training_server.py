#!/usr/bin/env python3
"""
KESTREL RL Training Server
Proximal Policy Optimization (PPO) agent for multipath packet scheduling

This server receives network state from Android, computes scheduling actions,
and updates the policy based on rewards from the environment.
"""

import socket
import json
import threading
import time
import os
import numpy as np
from datetime import datetime
from collections import deque
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# ============================================================================
# Configuration
# ============================================================================

SERVER_HOST = '0.0.0.0'
SERVER_PORT = 6000

# PPO Hyperparameters
LEARNING_RATE = 3e-4
GAMMA = 0.99  # Discount factor
GAE_LAMBDA = 0.95  # GAE parameter
PPO_CLIP_EPSILON = 0.2
PPO_EPOCHS = 10  # Number of optimization epochs per update
ENTROPY_COEF = 0.01  # Entropy bonus for exploration
VALUE_LOSS_COEF = 0.5
MAX_GRAD_NORM = 0.5

# State and action dimensions
STATE_DIM = 16
ACTION_DIM = 6

# Model saving
CHECKPOINT_INTERVAL = 10  # Save every N episodes
MODEL_DIR = "models"

# ============================================================================
# Neural Network Architecture
# ============================================================================

class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.
    Actor: Maps state -> action (mean and std for continuous actions)
    Critic: Maps state -> value estimate
    """

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM,
                 hidden_dim: int = 256):
        super(ActorCritic, self).__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        # Actor head (policy)
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Sigmoid()  # Actions are in [0, 1]
        )

        # Log std is a learnable parameter
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)

        # Actor output layer should have smaller weights
        nn.init.orthogonal_(self.actor_mean[-2].weight, gain=0.01)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            state: State tensor of shape (batch_size, state_dim)

        Returns:
            action_mean: Mean of action distribution
            action_std: Std of action distribution
            value: Value estimate
        """
        features = self.shared(state)

        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_log_std.clamp(-20, 2))
        value = self.critic(features)

        return action_mean, action_std, value

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from the policy.

        Args:
            state: State tensor
            deterministic: If True, return mean action (for evaluation)

        Returns:
            action: Sampled action
            log_prob: Log probability of the action
            value: Value estimate
        """
        action_mean, action_std, value = self.forward(state)

        if deterministic:
            action = action_mean
            log_prob = torch.zeros(1)
        else:
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            action = torch.clamp(action, 0, 1)  # Ensure actions in [0, 1]
            log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob, value

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions given states (for PPO update).

        Args:
            states: Batch of states
            actions: Batch of actions

        Returns:
            log_probs: Log probabilities of actions
            values: Value estimates
            entropy: Entropy of the policy
        """
        action_mean, action_std, values = self.forward(states)

        dist = Normal(action_mean, action_std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_probs, values.squeeze(-1), entropy


# ============================================================================
# PPO Agent
# ============================================================================

class PPOAgent:
    """
    Proximal Policy Optimization agent for multipath scheduling.
    """

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM,
                 lr: float = LEARNING_RATE, device: str = None):

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Episode buffer
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

        # Training statistics
        self.episode_count = 0
        self.best_avg_reward = float('-inf')
        self.training_log = []

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """
        Select action given current state.

        Args:
            state: Current state (16D array)
            deterministic: If True, use mean action

        Returns:
            action: Action array (6D)
            log_prob: Log probability of action
            value: Value estimate
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob, value = self.policy.get_action(state_tensor, deterministic)

        action_np = action.cpu().numpy().squeeze()

        # Normalize weights (first 4 elements should sum to 1)
        weights = action_np[:4]
        weights_sum = weights.sum()
        if weights_sum > 0:
            action_np[:4] = weights / weights_sum
        else:
            action_np[:4] = np.array([0.25, 0.25, 0.25, 0.25])

        return action_np, log_prob.item(), value.item()

    def store_transition(self, state: np.ndarray, action: np.ndarray,
                         log_prob: float, reward: float, value: float, done: bool):
        """Store transition in episode buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_gae(self, next_value: float = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            next_value: Bootstrap value for final state

        Returns:
            returns: Discounted returns
            advantages: GAE advantages
        """
        rewards = np.array(self.rewards)
        values = np.array(self.values + [next_value])
        dones = np.array(self.dones + [True])

        advantages = np.zeros_like(rewards)
        gae = 0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + GAMMA * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values[:-1]

        return returns, advantages

    def update(self) -> Dict[str, float]:
        """
        Perform PPO update using collected episode data.

        Returns:
            Dictionary with loss metrics
        """
        if len(self.states) == 0:
            return {}

        # Compute returns and advantages
        returns, advantages = self.compute_gae()

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        # PPO update
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        for _ in range(PPO_EPOCHS):
            # Evaluate current policy
            log_probs, values, entropy = self.policy.evaluate_actions(states, actions)

            # Policy loss with clipping
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - PPO_CLIP_EPSILON, 1 + PPO_CLIP_EPSILON) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = nn.MSELoss()(values, returns)

            # Entropy bonus
            entropy_loss = -entropy.mean()

            # Total loss
            loss = policy_loss + VALUE_LOSS_COEF * value_loss + ENTROPY_COEF * entropy_loss

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), MAX_GRAD_NORM)
            self.optimizer.step()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()

        # Clear episode buffer
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

        return {
            'loss': total_loss / PPO_EPOCHS,
            'policy_loss': total_policy_loss / PPO_EPOCHS,
            'value_loss': total_value_loss / PPO_EPOCHS,
            'entropy': total_entropy / PPO_EPOCHS
        }

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_count': self.episode_count,
            'best_avg_reward': self.best_avg_reward,
            'training_log': self.training_log
        }, path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_count = checkpoint.get('episode_count', 0)
        self.best_avg_reward = checkpoint.get('best_avg_reward', float('-inf'))
        self.training_log = checkpoint.get('training_log', [])
        print(f"Model loaded from {path}")


# ============================================================================
# Reward Functions
# ============================================================================

def calculate_video_streaming_reward(metrics: Dict) -> float:
    """
    Calculate reward for video streaming intent.
    Prioritizes low delay, low jitter, and stability.

    Args:
        metrics: Dictionary with p95_delay, p95_jitter, loss_rate, throughput, stall_count

    Returns:
        Reward value
    """
    p95_delay = metrics.get('p95_delay', 100)
    p95_jitter = metrics.get('p95_jitter', 20)
    loss_rate = metrics.get('loss_rate', 0.01)
    throughput = metrics.get('throughput', 5)
    stall_count = metrics.get('stall_count', 0)

    reward = (
        -0.3 * (p95_delay / 100.0)      # Penalize high delay
        - 0.3 * (p95_jitter / 50.0)     # Penalize jitter (KEY for video!)
        - 0.2 * (loss_rate * 10.0)      # Penalize packet loss
        - 0.1 * (stall_count * 5.0)     # Heavily penalize stalls
        + 0.1 * (throughput / 5.0)      # Small reward for throughput
    )

    return reward


def calculate_file_transfer_reward(metrics: Dict) -> float:
    """
    Calculate reward for file transfer intent.
    Prioritizes high throughput and completion speed.

    Args:
        metrics: Dictionary with throughput, loss_rate, bytes_sent, completion_time

    Returns:
        Reward value
    """
    throughput = metrics.get('throughput', 5)
    loss_rate = metrics.get('loss_rate', 0.01)
    bytes_sent = metrics.get('bytes_sent', 0)
    completion_time = metrics.get('completion_time', 1)
    done = metrics.get('done', False)

    # Completion bonus
    completion_bonus = 0
    if bytes_sent > 0 and done:
        ideal_time = (bytes_sent * 8 / 1e6) / 50.0  # Ideal at 50 Mbps
        completion_bonus = min(ideal_time / max(completion_time, 0.001), 2.0)

    reward = (
        + 0.6 * (throughput / 20.0)     # MAXIMIZE throughput (KEY!)
        - 0.2 * (loss_rate * 15.0)      # Penalize loss
        + 0.2 * completion_bonus        # Bonus for fast finish
    )

    return reward


def calculate_reward(metrics: Dict, intent: str) -> float:
    """
    Calculate reward based on intent type.

    Args:
        metrics: QoS metrics from Android
        intent: 'video_streaming' or 'file_transfer'

    Returns:
        Reward value
    """
    if intent == 'video_streaming':
        return calculate_video_streaming_reward(metrics)
    elif intent == 'file_transfer':
        return calculate_file_transfer_reward(metrics)
    else:
        # Default: weighted average
        return 0.5 * calculate_video_streaming_reward(metrics) + \
               0.5 * calculate_file_transfer_reward(metrics)


# ============================================================================
# State Processing
# ============================================================================

def state_to_array(state_dict: Dict) -> np.ndarray:
    """
    Convert state dictionary from Android to normalized numpy array.

    Args:
        state_dict: Dictionary with wifi, cellular, and intent fields

    Returns:
        Normalized state array (16D)
    """
    wifi = state_dict.get('wifi', {})
    cellular = state_dict.get('cellular', {})
    intent = state_dict.get('intent', 'video_streaming')

    # WiFi metrics (7D)
    wifi_state = [
        wifi.get('srtt', 50) / 200.0,           # Normalized smoothed RTT
        wifi.get('jitter', 10) / 50.0,          # Normalized jitter
        wifi.get('burst', 0.2),                 # Burst indicator (already 0-1)
        wifi.get('loss', 0.01),                 # Loss rate (already 0-1)
        wifi.get('throughput', 20) / 100.0,     # Normalized throughput
        wifi.get('queue_depth', 500) / 10000.0, # Normalized queue size
        1.0 if wifi.get('available', True) else 0.0
    ]

    # Cellular metrics (7D)
    cellular_state = [
        cellular.get('srtt', 80) / 200.0,
        cellular.get('jitter', 15) / 50.0,
        cellular.get('burst', 0.3),
        cellular.get('loss', 0.02),
        cellular.get('throughput', 15) / 100.0,
        cellular.get('queue_depth', 300) / 10000.0,
        1.0 if cellular.get('available', True) else 0.0
    ]

    # Intent (2D one-hot)
    intent_state = [
        1.0 if intent == 'video_streaming' else 0.0,
        1.0 if intent == 'file_transfer' else 0.0
    ]

    # Combine and clip to [0, 1]
    state = np.array(wifi_state + cellular_state + intent_state, dtype=np.float32)
    state = np.clip(state, 0, 1)

    return state


def action_to_dict(action: np.ndarray) -> Dict:
    """
    Convert action array to dictionary for sending to Android.

    Args:
        action: Action array (6D)

    Returns:
        Action dictionary
    """
    return {
        'weight_delay': float(action[0]),
        'weight_jitter': float(action[1]),
        'weight_loss': float(action[2]),
        'weight_throughput': float(action[3]),
        'use_wifi': float(action[4]),
        'use_duplication': float(action[5])
    }


# ============================================================================
# RL Training Server
# ============================================================================

class RLTrainingServer:
    """
    TCP server that handles RL training communication with Android app.
    """

    def __init__(self, host: str = SERVER_HOST, port: int = SERVER_PORT):
        self.host = host
        self.port = port
        self.agent = PPOAgent()
        self.running = False

        # Episode tracking
        self.current_episode = 0
        self.episode_rewards = []
        self.episode_steps = 0
        self.episode_start_time = None
        self.current_intent = 'video_streaming'
        self.wifi_packets = 0
        self.cell_packets = 0

        # Recent state/action for reward assignment
        self.last_state = None
        self.last_action = None
        self.last_log_prob = None
        self.last_value = None

        # Ensure model directory exists
        os.makedirs(MODEL_DIR, exist_ok=True)

    def handle_get_action(self, data: Dict) -> Dict:
        """
        Handle get_action request from Android.

        Args:
            data: Request data with state information

        Returns:
            Response with action
        """
        state_dict = data.get('state', {})
        self.current_intent = state_dict.get('intent', 'video_streaming')

        # Convert state to array
        state = state_to_array(state_dict)

        # Get action from policy
        action, log_prob, value = self.agent.select_action(state)

        # Store for later reward assignment
        self.last_state = state
        self.last_action = action
        self.last_log_prob = log_prob
        self.last_value = value

        # Convert action to dictionary
        action_dict = action_to_dict(action)

        return {'action': action_dict}

    def handle_report_reward(self, data: Dict) -> Dict:
        """
        Handle report_reward request from Android.

        Args:
            data: Request data with metrics

        Returns:
            Acknowledgment response
        """
        metrics = data.get('metrics', {})
        intent = data.get('intent', self.current_intent)
        done = data.get('done', False)

        # Track packet distribution
        self.wifi_packets += metrics.get('wifi_packets', 0)
        self.cell_packets += metrics.get('cell_packets', 0)

        # Calculate reward
        reward = calculate_reward(metrics, intent)
        self.episode_rewards.append(reward)
        self.episode_steps += 1

        # Store transition if we have a previous state
        if self.last_state is not None:
            self.agent.store_transition(
                self.last_state,
                self.last_action,
                self.last_log_prob,
                reward,
                self.last_value,
                done
            )

        return {
            'status': 'ok',
            'reward': reward,
            'episode_reward': sum(self.episode_rewards)
        }

    def handle_episode_done(self, data: Dict) -> Dict:
        """
        Handle episode_done request from Android.

        Args:
            data: Request data

        Returns:
            Episode summary response
        """
        self.current_episode += 1

        # Calculate episode statistics
        total_reward = sum(self.episode_rewards)
        avg_reward = total_reward / max(len(self.episode_rewards), 1)
        total_packets = self.wifi_packets + self.cell_packets
        wifi_usage = (self.wifi_packets / max(total_packets, 1)) * 100
        cell_usage = (self.cell_packets / max(total_packets, 1)) * 100

        duration = time.time() - self.episode_start_time if self.episode_start_time else 0

        print(f"\nEpisode {self.current_episode} finished:")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Steps: {self.episode_steps}")
        print(f"  Avg reward: {avg_reward:.4f}")
        print(f"  WiFi usage: {wifi_usage:.1f}% | Cellular usage: {cell_usage:.1f}%")

        # Update policy
        loss_info = self.agent.update()
        if loss_info:
            print(f"Policy updated: loss={loss_info['loss']:.4f}, "
                  f"policy_loss={loss_info['policy_loss']:.4f}, "
                  f"value_loss={loss_info['value_loss']:.4f}")

        # Track if this is the best episode
        is_best = avg_reward > self.agent.best_avg_reward
        if is_best:
            self.agent.best_avg_reward = avg_reward
            best_path = os.path.join(MODEL_DIR, "kestrel_model_BEST.pth")
            self.agent.save(best_path)
            print(f"NEW BEST MODEL! Reward: {avg_reward:.4f} (Episode {self.current_episode})")

        # Save checkpoint
        if self.current_episode % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = os.path.join(MODEL_DIR, f"kestrel_model_episode_{self.current_episode}.pth")
            self.agent.save(checkpoint_path)

        # Log episode
        log_entry = {
            'episode': self.current_episode,
            'total_reward': total_reward,
            'avg_reward': avg_reward,
            'steps': self.episode_steps,
            'duration_seconds': duration,
            'intent': self.current_intent,
            'wifi_usage_percent': wifi_usage,
            'cell_usage_percent': cell_usage,
            'is_best': is_best
        }
        self.agent.training_log.append(log_entry)

        # Save training log
        log_path = os.path.join(MODEL_DIR, "training_log.json")
        with open(log_path, 'w') as f:
            json.dump(self.agent.training_log, f, indent=2)

        # Reset episode counters
        self.episode_rewards = []
        self.episode_steps = 0
        self.episode_start_time = time.time()
        self.wifi_packets = 0
        self.cell_packets = 0
        self.last_state = None
        self.last_action = None

        self.agent.episode_count = self.current_episode

        return {
            'status': 'ok',
            'episode': self.current_episode,
            'total_reward': total_reward,
            'avg_reward': avg_reward,
            'is_best': is_best
        }

    def handle_client(self, conn: socket.socket, addr: Tuple[str, int]):
        """
        Handle a connected client.

        Args:
            conn: Client socket
            addr: Client address
        """
        print(f"\nConnection from {addr}")
        self.episode_start_time = time.time()

        buffer = ""

        try:
            while self.running:
                # Receive data
                data = conn.recv(4096)
                if not data:
                    break

                # Add to buffer and process complete messages
                buffer += data.decode('utf-8')

                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if not line.strip():
                        continue

                    try:
                        request = json.loads(line)
                        msg_type = request.get('type', '')

                        if msg_type == 'get_action':
                            response = self.handle_get_action(request)
                        elif msg_type == 'report_reward':
                            response = self.handle_report_reward(request)
                        elif msg_type == 'episode_done':
                            response = self.handle_episode_done(request)
                        else:
                            response = {'error': f'Unknown message type: {msg_type}'}

                        # Send response
                        response_str = json.dumps(response) + '\n'
                        conn.sendall(response_str.encode('utf-8'))

                    except json.JSONDecodeError as e:
                        print(f"JSON decode error: {e}")
                        error_response = json.dumps({'error': str(e)}) + '\n'
                        conn.sendall(error_response.encode('utf-8'))

        except Exception as e:
            print(f"Client handler error: {e}")
        finally:
            print(f"Client {addr} disconnected")
            conn.close()

    def start(self):
        """Start the RL training server."""
        self.running = True

        # Create server socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.host, self.port))
        server_socket.listen(1)
        server_socket.settimeout(1.0)

        print("=" * 60)
        print("KESTREL RL Training Server")
        print("=" * 60)
        print(f"RL Training Server started on {self.host}:{self.port}")
        print("Waiting for Android app to connect...")
        print("=" * 60)

        try:
            while self.running:
                try:
                    conn, addr = server_socket.accept()
                    # Handle client in the main thread (one client at a time)
                    self.handle_client(conn, addr)
                except socket.timeout:
                    continue

        except KeyboardInterrupt:
            print("\n\nShutting down server...")
        finally:
            self.running = False
            server_socket.close()

            # Save final model
            if self.current_episode > 0:
                final_path = os.path.join(MODEL_DIR, "kestrel_model_FINAL.pth")
                self.agent.save(final_path)

            print("\nServer stopped.")
            print(f"Total episodes: {self.current_episode}")
            print(f"Best avg reward: {self.agent.best_avg_reward:.4f}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("  KESTREL - On-Device RL for Multipath Scheduling")
    print("  PPO Training Server")
    print("=" * 60 + "\n")

    server = RLTrainingServer(host=SERVER_HOST, port=SERVER_PORT)
    server.start()


if __name__ == "__main__":
    main()
