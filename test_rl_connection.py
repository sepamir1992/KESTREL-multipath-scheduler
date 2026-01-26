#!/usr/bin/env python3
"""
KESTREL RL Connection Test Script

This script simulates an Android client to test the RL training server.
Run this after starting rl_training_server.py to verify everything works.
"""

import socket
import json
import time
import random
import argparse


def send_message(sock: socket.socket, message: dict) -> dict:
    """Send a message and receive response."""
    msg_str = json.dumps(message) + '\n'
    sock.sendall(msg_str.encode('utf-8'))

    # Receive response
    response = ""
    while '\n' not in response:
        data = sock.recv(4096)
        if not data:
            break
        response += data.decode('utf-8')

    return json.loads(response.strip())


def generate_fake_state(intent: str = 'video_streaming') -> dict:
    """Generate a fake network state for testing."""
    return {
        'wifi': {
            'srtt': random.uniform(15, 50),
            'jitter': random.uniform(2, 15),
            'burst': random.uniform(0.1, 0.4),
            'loss': random.uniform(0.0, 0.02),
            'throughput': random.uniform(30, 80),
            'queue_depth': random.randint(500, 2000),
            'available': True
        },
        'cellular': {
            'srtt': random.uniform(40, 120),
            'jitter': random.uniform(5, 30),
            'burst': random.uniform(0.2, 0.6),
            'loss': random.uniform(0.0, 0.05),
            'throughput': random.uniform(10, 40),
            'queue_depth': random.randint(200, 1000),
            'available': True
        },
        'intent': intent
    }


def generate_fake_metrics() -> dict:
    """Generate fake QoS metrics for testing."""
    return {
        'p95_delay': random.uniform(20, 80),
        'p95_jitter': random.uniform(3, 20),
        'loss_rate': random.uniform(0.0, 0.03),
        'throughput': random.uniform(20, 60),
        'stall_count': random.randint(0, 1),
        'bytes_sent': random.randint(10000, 20000),
        'completion_time': random.uniform(1.0, 3.0),
        'wifi_packets': random.randint(5, 10),
        'cell_packets': random.randint(0, 5)
    }


def run_test_episode(sock: socket.socket, episode_num: int, steps: int = 10,
                     intent: str = 'video_streaming'):
    """Run a single test episode."""
    print(f"\n{'='*50}")
    print(f"Episode {episode_num} - Intent: {intent}")
    print(f"{'='*50}")

    total_reward = 0

    for step in range(steps):
        # Step 1: Get action
        state = generate_fake_state(intent)
        request = {
            'type': 'get_action',
            'state': state
        }

        response = send_message(sock, request)
        action = response.get('action', {})

        print(f"\nStep {step + 1}/{steps}")
        print(f"  State: WiFi RTT={state['wifi']['srtt']:.1f}ms, "
              f"Cell RTT={state['cellular']['srtt']:.1f}ms")
        print(f"  Action: weights=[{action.get('weight_delay', 0):.2f}, "
              f"{action.get('weight_jitter', 0):.2f}, "
              f"{action.get('weight_loss', 0):.2f}, "
              f"{action.get('weight_throughput', 0):.2f}], "
              f"use_wifi={action.get('use_wifi', 0):.2f}")

        # Step 2: Report reward
        metrics = generate_fake_metrics()
        done = (step == steps - 1)

        reward_request = {
            'type': 'report_reward',
            'metrics': metrics,
            'intent': intent,
            'done': done
        }

        reward_response = send_message(sock, reward_request)
        reward = reward_response.get('reward', 0)
        total_reward += reward

        print(f"  Metrics: p95_delay={metrics['p95_delay']:.1f}ms, "
              f"throughput={metrics['throughput']:.1f}Mbps")
        print(f"  Reward: {reward:.4f} (total: {total_reward:.4f})")

        time.sleep(0.1)  # Small delay between steps

    # Episode done
    done_request = {'type': 'episode_done'}
    done_response = send_message(sock, done_request)

    print(f"\nEpisode {episode_num} complete!")
    print(f"  Total reward: {done_response.get('total_reward', 0):.4f}")
    print(f"  Avg reward: {done_response.get('avg_reward', 0):.4f}")
    print(f"  Is best: {done_response.get('is_best', False)}")

    return done_response


def main():
    parser = argparse.ArgumentParser(description='Test KESTREL RL Training Server')
    parser.add_argument('--host', default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=6000, help='Server port')
    parser.add_argument('--episodes', type=int, default=5, help='Number of test episodes')
    parser.add_argument('--steps', type=int, default=10, help='Steps per episode')
    parser.add_argument('--intent', default='video_streaming',
                        choices=['video_streaming', 'file_transfer'],
                        help='Intent type for testing')
    args = parser.parse_args()

    print("=" * 60)
    print("KESTREL RL Connection Test")
    print("=" * 60)
    print(f"Connecting to {args.host}:{args.port}...")

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((args.host, args.port))
        print("Connected!")

        # Run test episodes
        for ep in range(1, args.episodes + 1):
            # Alternate intents for variety
            intent = args.intent if ep % 2 == 1 else (
                'file_transfer' if args.intent == 'video_streaming' else 'video_streaming'
            )
            run_test_episode(sock, ep, args.steps, intent)
            time.sleep(0.5)

        print("\n" + "=" * 60)
        print("All test episodes completed successfully!")
        print("=" * 60)

    except ConnectionRefusedError:
        print(f"ERROR: Could not connect to server at {args.host}:{args.port}")
        print("Make sure rl_training_server.py is running first.")
        return 1
    except Exception as e:
        print(f"ERROR: {e}")
        return 1
    finally:
        sock.close()

    return 0


if __name__ == "__main__":
    exit(main())
