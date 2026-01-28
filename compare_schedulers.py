#!/usr/bin/env python3
"""
KESTREL Scheduler Comparison Tool

This script analyzes and compares performance between:
- RL-based scheduler (KESTREL)
- MinRTT baseline
- Round Robin baseline
- Weighted Round Robin baseline

Usage:
    python compare_schedulers.py --rl-log models/training_log.json --baseline-dir baseline_logs/
    python compare_schedulers.py --simulate-baseline --rl-log models/training_log.json

Author: KESTREL Team
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import statistics
from pathlib import Path


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""
    episode: int
    scheduler: str
    total_reward: float
    avg_reward: float
    steps: int
    duration_seconds: float
    wifi_usage_percent: float
    cell_usage_percent: float
    avg_p95_delay: Optional[float] = None
    avg_p95_jitter: Optional[float] = None
    avg_loss_rate: Optional[float] = None
    avg_throughput: Optional[float] = None


@dataclass
class SchedulerStats:
    """Aggregate statistics for a scheduler."""
    name: str
    episodes: int
    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    median_reward: float
    mean_wifi_usage: float
    mean_steps: float
    best_episode: int
    best_reward: float


# ============================================================================
# Reward Calculation (for baseline comparison)
# ============================================================================

def calculate_simulated_reward_v3(
    p95_delay: float,
    p95_jitter: float,
    loss_rate: float,
    throughput: float,
    stall_count: int = 0,
    wifi_delay: float = 50.0,
    wifi_jitter: float = 10.0,
    cell_delay: float = 80.0,
    cell_jitter: float = 15.0
) -> float:
    """
    Calculate what the V3 hybrid reward would be for given metrics.

    This allows fair comparison between RL and baselines by calculating
    what reward a baseline would have received under the RL reward function.

    Args:
        p95_delay: 95th percentile delay achieved (ms)
        p95_jitter: 95th percentile jitter achieved (ms)
        loss_rate: Packet loss rate (0-1)
        throughput: Throughput achieved (Mbps)
        stall_count: Number of stalls
        wifi_delay: WiFi path delay at decision time (ms)
        wifi_jitter: WiFi path jitter at decision time (ms)
        cell_delay: Cellular path delay at decision time (ms)
        cell_jitter: Cellular path jitter at decision time (ms)

    Returns:
        Simulated reward value
    """
    # ABSOLUTE COMPONENT (40%)
    absolute_reward = (
        -0.12 * (p95_delay / 100.0)
        - 0.12 * (p95_jitter / 50.0)
        + 0.08 * (throughput / 10.0)
        + 0.08 * max(0, 1.0 - loss_rate * 20)
    )

    # RELATIVE COMPONENT (60%)
    min_delay = min(wifi_delay, cell_delay)
    min_jitter = min(wifi_jitter, cell_jitter)

    relative_delay = max(0, p95_delay - min_delay)
    relative_jitter = max(0, p95_jitter - min_jitter)

    relative_reward = (
        -0.20 * (relative_delay / 75.0)
        - 0.20 * (relative_jitter / 40.0)
    )

    # FIXED PENALTIES (20%)
    fixed_penalties = (
        - 0.10 * (loss_rate * 10.0)
        - 0.10 * (stall_count * 3.0)
    )

    return absolute_reward + relative_reward + fixed_penalties


# ============================================================================
# Data Loading
# ============================================================================

def load_rl_training_log(path: str) -> List[EpisodeMetrics]:
    """Load RL training log from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)

    episodes = []
    for entry in data:
        episodes.append(EpisodeMetrics(
            episode=entry['episode'],
            scheduler='rl',
            total_reward=entry['total_reward'],
            avg_reward=entry['avg_reward'],
            steps=entry['steps'],
            duration_seconds=entry['duration_seconds'],
            wifi_usage_percent=entry['wifi_usage_percent'],
            cell_usage_percent=entry['cell_usage_percent']
        ))

    return episodes


def load_baseline_csv(path: str) -> List[EpisodeMetrics]:
    """Load baseline CSV log file."""
    # TODO: Implement when baseline logging is added to Android client
    # For now, return empty list
    print(f"Warning: Baseline CSV loading not yet implemented for {path}")
    return []


# ============================================================================
# Statistical Analysis
# ============================================================================

def calculate_stats(episodes: List[EpisodeMetrics], name: str) -> SchedulerStats:
    """Calculate aggregate statistics for a scheduler."""
    if not episodes:
        return SchedulerStats(
            name=name, episodes=0, mean_reward=0, std_reward=0,
            min_reward=0, max_reward=0, median_reward=0,
            mean_wifi_usage=0, mean_steps=0, best_episode=0, best_reward=0
        )

    rewards = [e.avg_reward for e in episodes]
    wifi_usages = [e.wifi_usage_percent for e in episodes]
    steps = [e.steps for e in episodes]

    best_idx = rewards.index(max(rewards))

    return SchedulerStats(
        name=name,
        episodes=len(episodes),
        mean_reward=statistics.mean(rewards),
        std_reward=statistics.stdev(rewards) if len(rewards) > 1 else 0,
        min_reward=min(rewards),
        max_reward=max(rewards),
        median_reward=statistics.median(rewards),
        mean_wifi_usage=statistics.mean(wifi_usages),
        mean_steps=statistics.mean(steps),
        best_episode=episodes[best_idx].episode,
        best_reward=max(rewards)
    )


def welch_t_test(sample1: List[float], sample2: List[float]) -> Tuple[float, float]:
    """
    Perform Welch's t-test (unequal variance t-test).

    Returns:
        Tuple of (t-statistic, p-value approximation)
    """
    n1, n2 = len(sample1), len(sample2)
    if n1 < 2 or n2 < 2:
        return 0.0, 1.0

    mean1, mean2 = statistics.mean(sample1), statistics.mean(sample2)
    var1, var2 = statistics.variance(sample1), statistics.variance(sample2)

    # Welch's t-statistic
    se = ((var1 / n1) + (var2 / n2)) ** 0.5
    if se == 0:
        return 0.0, 1.0

    t_stat = (mean1 - mean2) / se

    # Welch-Satterthwaite degrees of freedom
    df_num = ((var1 / n1) + (var2 / n2)) ** 2
    df_den = ((var1 / n1) ** 2 / (n1 - 1)) + ((var2 / n2) ** 2 / (n2 - 1))
    df = df_num / df_den if df_den > 0 else 1

    # Approximate p-value using normal distribution for large df
    # For more accurate p-value, use scipy.stats.t.sf
    import math
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2))))

    return t_stat, p_value


# ============================================================================
# Comparison and Reporting
# ============================================================================

def compare_schedulers(
    rl_episodes: List[EpisodeMetrics],
    minrtt_episodes: List[EpisodeMetrics],
    wrr_episodes: List[EpisodeMetrics]
) -> Dict:
    """
    Compare scheduler performance and return statistical analysis.
    """
    rl_stats = calculate_stats(rl_episodes, 'RL (KESTREL)')
    minrtt_stats = calculate_stats(minrtt_episodes, 'MinRTT')
    wrr_stats = calculate_stats(wrr_episodes, 'Weighted RR')

    results = {
        'rl': rl_stats,
        'minrtt': minrtt_stats,
        'wrr': wrr_stats,
        'comparisons': {}
    }

    # Statistical comparisons
    if rl_episodes and minrtt_episodes:
        rl_rewards = [e.avg_reward for e in rl_episodes]
        minrtt_rewards = [e.avg_reward for e in minrtt_episodes]
        t_stat, p_value = welch_t_test(rl_rewards, minrtt_rewards)
        results['comparisons']['rl_vs_minrtt'] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'rl_better': rl_stats.mean_reward > minrtt_stats.mean_reward
        }

    if rl_episodes and wrr_episodes:
        rl_rewards = [e.avg_reward for e in rl_episodes]
        wrr_rewards = [e.avg_reward for e in wrr_episodes]
        t_stat, p_value = welch_t_test(rl_rewards, wrr_rewards)
        results['comparisons']['rl_vs_wrr'] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'rl_better': rl_stats.mean_reward > wrr_stats.mean_reward
        }

    return results


def generate_report(results: Dict) -> str:
    """Generate a text report from comparison results."""
    lines = []
    lines.append("=" * 70)
    lines.append("KESTREL SCHEDULER COMPARISON REPORT")
    lines.append("=" * 70)
    lines.append("")

    # Summary table
    lines.append("SUMMARY STATISTICS")
    lines.append("-" * 70)
    lines.append(f"{'Scheduler':<20} {'Episodes':<10} {'Mean Reward':<15} {'Std Dev':<12} {'WiFi %':<10}")
    lines.append("-" * 70)

    for key in ['rl', 'minrtt', 'wrr']:
        stats = results.get(key)
        if stats and stats.episodes > 0:
            lines.append(
                f"{stats.name:<20} {stats.episodes:<10} {stats.mean_reward:<15.4f} "
                f"{stats.std_reward:<12.4f} {stats.mean_wifi_usage:<10.1f}"
            )

    lines.append("")

    # Best models
    lines.append("BEST EPISODES")
    lines.append("-" * 70)
    for key in ['rl', 'minrtt', 'wrr']:
        stats = results.get(key)
        if stats and stats.episodes > 0:
            lines.append(f"{stats.name}: Episode {stats.best_episode} (reward: {stats.best_reward:.4f})")

    lines.append("")

    # Statistical comparisons
    lines.append("STATISTICAL COMPARISONS")
    lines.append("-" * 70)

    comparisons = results.get('comparisons', {})

    if 'rl_vs_minrtt' in comparisons:
        comp = comparisons['rl_vs_minrtt']
        sig = "YES (p < 0.05)" if comp['significant'] else "NO (p >= 0.05)"
        better = "RL" if comp['rl_better'] else "MinRTT"
        lines.append(f"RL vs MinRTT:")
        lines.append(f"  t-statistic: {comp['t_statistic']:.4f}")
        lines.append(f"  p-value: {comp['p_value']:.4f}")
        lines.append(f"  Significant difference: {sig}")
        lines.append(f"  Better performer: {better}")
        lines.append("")

    if 'rl_vs_wrr' in comparisons:
        comp = comparisons['rl_vs_wrr']
        sig = "YES (p < 0.05)" if comp['significant'] else "NO (p >= 0.05)"
        better = "RL" if comp['rl_better'] else "Weighted RR"
        lines.append(f"RL vs Weighted RR:")
        lines.append(f"  t-statistic: {comp['t_statistic']:.4f}")
        lines.append(f"  p-value: {comp['p_value']:.4f}")
        lines.append(f"  Significant difference: {sig}")
        lines.append(f"  Better performer: {better}")
        lines.append("")

    # Recommendation
    lines.append("RECOMMENDATION")
    lines.append("-" * 70)

    rl_stats = results.get('rl')
    minrtt_stats = results.get('minrtt')

    if rl_stats and rl_stats.episodes > 0:
        if minrtt_stats and minrtt_stats.episodes > 0:
            improvement = ((rl_stats.mean_reward - minrtt_stats.mean_reward) /
                          abs(minrtt_stats.mean_reward)) * 100

            rl_vs_minrtt = comparisons.get('rl_vs_minrtt', {})

            if improvement > 20 and rl_vs_minrtt.get('significant', False):
                lines.append("DEPLOY RL: Significant improvement over baselines (>20%)")
            elif improvement > 10:
                lines.append("CONTINUE TRAINING: Moderate improvement, need more data")
            elif abs(improvement) <= 10:
                lines.append("USE MINRTT: RL provides no significant benefit, use simpler baseline")
            else:
                lines.append("INVESTIGATE: RL performing worse than baseline, review reward function")
        else:
            lines.append("COLLECT BASELINE DATA: Need MinRTT/WRR comparison data")
    else:
        lines.append("NO RL DATA: Run RL training first")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def analyze_convergence(episodes: List[EpisodeMetrics], window: int = 50) -> Dict:
    """
    Analyze training convergence.

    Args:
        episodes: List of episode metrics
        window: Window size for rolling statistics

    Returns:
        Convergence analysis results
    """
    if len(episodes) < window:
        return {'converged': False, 'reason': 'Not enough episodes'}

    rewards = [e.avg_reward for e in episodes]

    # Check last window vs previous window
    recent = rewards[-window:]
    previous = rewards[-2*window:-window] if len(rewards) >= 2*window else rewards[:window]

    recent_mean = statistics.mean(recent)
    previous_mean = statistics.mean(previous)
    recent_std = statistics.stdev(recent) if len(recent) > 1 else 0

    # Convergence criteria
    improvement = abs(recent_mean - previous_mean) / abs(previous_mean) if previous_mean != 0 else 0
    variance_low = recent_std < 0.05

    converged = improvement < 0.02 and variance_low

    return {
        'converged': converged,
        'recent_mean': recent_mean,
        'previous_mean': previous_mean,
        'improvement_rate': improvement,
        'recent_std': recent_std,
        'variance_low': variance_low
    }


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='KESTREL Scheduler Comparison Tool')
    parser.add_argument('--rl-log', type=str, default='models/training_log.json',
                        help='Path to RL training log JSON file')
    parser.add_argument('--baseline-dir', type=str, default='baseline_logs',
                        help='Directory containing baseline CSV logs')
    parser.add_argument('--output', type=str, default='comparison_report.txt',
                        help='Output file for comparison report')
    parser.add_argument('--convergence-window', type=int, default=50,
                        help='Window size for convergence analysis')

    args = parser.parse_args()

    print("KESTREL Scheduler Comparison Tool")
    print("=" * 50)

    # Load RL data
    rl_episodes = []
    if os.path.exists(args.rl_log):
        print(f"Loading RL training log from {args.rl_log}...")
        rl_episodes = load_rl_training_log(args.rl_log)
        print(f"  Loaded {len(rl_episodes)} episodes")
    else:
        print(f"Warning: RL log not found at {args.rl_log}")

    # Load baseline data (when available)
    minrtt_episodes = []
    wrr_episodes = []

    if os.path.exists(args.baseline_dir):
        for filename in os.listdir(args.baseline_dir):
            filepath = os.path.join(args.baseline_dir, filename)
            if 'minrtt' in filename.lower():
                minrtt_episodes.extend(load_baseline_csv(filepath))
            elif 'wrr' in filename.lower() or 'weighted' in filename.lower():
                wrr_episodes.extend(load_baseline_csv(filepath))

    # Run comparison
    print("\nRunning comparison analysis...")
    results = compare_schedulers(rl_episodes, minrtt_episodes, wrr_episodes)

    # Convergence analysis
    if rl_episodes:
        print("\nAnalyzing RL convergence...")
        convergence = analyze_convergence(rl_episodes, args.convergence_window)
        results['convergence'] = convergence

        if convergence['converged']:
            print("  RL training appears to have CONVERGED")
        else:
            print("  RL training has NOT converged yet")
        print(f"  Recent mean reward: {convergence['recent_mean']:.4f}")
        print(f"  Recent std dev: {convergence['recent_std']:.4f}")

    # Generate report
    report = generate_report(results)
    print("\n" + report)

    # Save report
    with open(args.output, 'w') as f:
        f.write(report)
    print(f"\nReport saved to {args.output}")

    # Save detailed results as JSON
    json_output = args.output.replace('.txt', '.json')
    with open(json_output, 'w') as f:
        # Convert dataclasses to dicts for JSON serialization
        json_results = {
            'rl': vars(results['rl']) if results['rl'].episodes > 0 else None,
            'minrtt': vars(results['minrtt']) if results['minrtt'].episodes > 0 else None,
            'wrr': vars(results['wrr']) if results['wrr'].episodes > 0 else None,
            'comparisons': results['comparisons'],
            'convergence': results.get('convergence')
        }
        json.dump(json_results, f, indent=2)
    print(f"Detailed results saved to {json_output}")


if __name__ == '__main__':
    main()
