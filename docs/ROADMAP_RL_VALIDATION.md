# KESTREL RL Validation Roadmap

**Document Version:** 1.0
**Created:** January 2026
**Status:** In Progress

---

## Executive Summary

This document outlines the systematic validation process required before deploying the KESTREL RL-based multipath scheduler in production. The goal is to ensure the RL approach provides measurable benefits over simple heuristic baselines.

---

## Current State Assessment

### What We Have

| Component | Status | Notes |
|-----------|--------|-------|
| RL Training Infrastructure | ✅ Complete | PPO agent, reward functions, episode management |
| MinRTT Baseline | ✅ Implemented | Simple RTT comparison |
| Round Robin Baseline | ✅ Implemented | Alternating path selection |
| RL Detailed Logging | ✅ Complete | CSV with 30+ columns |
| Baseline Detailed Logging | ❌ Missing | Only basic console output |
| Baseline Comparison Framework | ❌ Missing | No automated comparison |
| Production Deployment Strategy | ❌ Missing | No safety mechanisms |

### What We Need

1. **Equivalent logging for all schedulers** - Same metrics format for fair comparison
2. **Extended RL training** - 500+ episodes for convergence
3. **Automated comparison tools** - Scripts to analyze and compare results
4. **Hybrid reward function** - Balance exploration with performance
5. **Production safety mechanisms** - Fallback and monitoring

---

## Phase 1: Baseline Logging Enhancement

### Objective
Add comprehensive logging to MinRTT and Round Robin modes equivalent to RL logging.

### Required Changes

#### 1.1 Android Client Modifications

**File:** `MultipathClient/app/src/main/java/com/example/multipathclient/MainActivity.kt`

**Add baseline logging with same format as RL:**

```kotlin
// New: Baseline episode logging
data class BaselineEpisodeLog(
    val scheduler: String,           // "minrtt", "round_robin", "weighted_rr"
    val episode: Int,
    val duration_seconds: Double,
    val steps: Int,
    val total_packets: Int,
    val wifi_packets: Int,
    val cell_packets: Int,
    val wifi_usage_percent: Double,
    val cell_usage_percent: Double,
    val avg_p95_delay: Double,
    val avg_p95_jitter: Double,
    val avg_loss_rate: Double,
    val avg_throughput: Double,
    val total_bytes_sent: Long,
    val stall_count: Int
)
```

**Metrics to collect for baselines (same as RL):**

| Metric | Description | Collection Point |
|--------|-------------|------------------|
| `p95_delay` | 95th percentile RTT | Every 10 packets (step) |
| `p95_jitter` | 95th percentile jitter | Every 10 packets (step) |
| `loss_rate` | Packet loss ratio | Cumulative per episode |
| `throughput` | Mbps achieved | Rolling average |
| `wifi_usage_percent` | % packets via WiFi | End of episode |
| `stall_count` | Playback stalls (video) | Simulated based on delay |

#### 1.2 Baseline CSV Format

**File:** `/sdcard/Android/data/com.example.multipathclient/files/baseline_{scheduler}_{timestamp}.csv`

```csv
timestamp,scheduler,episode,step,
wifi_srtt,wifi_jitter,wifi_loss,wifi_throughput,
cell_srtt,cell_jitter,cell_loss,cell_throughput,
wifi_packets,cell_packets,p95_delay,p95_jitter,loss_rate,throughput,
simulated_reward
```

**Note:** `simulated_reward` = Calculate what reward would have been under RL reward function for fair comparison.

#### 1.3 Weighted Round Robin Implementation

**Current RR:** Alternates WiFi/Cellular every packet (50/50)

**Weighted RR Options:**

| Variant | Logic | Use Case |
|---------|-------|----------|
| WRR-70/30 | 70% WiFi, 30% Cellular | WiFi-preferred baseline |
| WRR-RTT | Weight inversely proportional to RTT | Adaptive weighted |
| WRR-Throughput | Weight proportional to throughput | Throughput-optimized |

**Recommended Implementation (WRR-RTT):**

```kotlin
// Weighted Round Robin based on inverse RTT
fun selectPathWeightedRR(): Boolean {
    val wifiWeight = 1.0 / (wifiRTT + 1.0)
    val cellWeight = 1.0 / (cellularRTT + 1.0)
    val totalWeight = wifiWeight + cellWeight
    val wifiProbability = wifiWeight / totalWeight
    return Math.random() < wifiProbability
}
```

---

## Phase 2: Extended RL Training

### Objective
Train RL agent for 500+ episodes to ensure convergence.

### Configuration

```python
# rl_training_server.py modifications
MAX_EPISODES = 500  # Increased from 100
CHECKPOINT_INTERVAL = 25  # More frequent checkpoints
CONVERGENCE_WINDOW = 50  # Episodes to check for convergence
CONVERGENCE_THRESHOLD = 0.02  # Reward variance threshold
```

### Convergence Criteria

Training is considered converged when:

1. **Reward plateau:** Rolling 50-episode average changes < 2%
2. **Policy stability:** Action variance decreases over time
3. **WiFi usage stability:** Usage % stabilizes within ±5%

### Training Monitoring

```
Episode Checkpoints:
├── Episode 100: Early checkpoint
├── Episode 200: Mid-training checkpoint
├── Episode 300: Late-training checkpoint
├── Episode 400: Pre-convergence checkpoint
└── Episode 500: Final checkpoint

Metrics to track:
├── Rolling average reward (50-episode window)
├── Reward standard deviation
├── WiFi usage trend
├── Policy entropy (exploration level)
└── Best model episode
```

---

## Phase 3: Reward Function Refinement

### Current Issues

1. **Relative-only penalties** lead to worse absolute performance
2. **No baseline reward** for good absolute QoS
3. **Harsh normalization** (50ms delay, 25ms jitter divisors)

### Proposed Hybrid Reward Function

```python
def calculate_video_streaming_reward_v3(metrics: Dict, path_metrics: Optional[Dict] = None) -> float:
    """
    Hybrid reward: 60% relative + 40% absolute performance
    """
    p95_delay = metrics.get('p95_delay', 100)
    p95_jitter = metrics.get('p95_jitter', 20)
    loss_rate = metrics.get('loss_rate', 0.01)
    throughput = metrics.get('throughput', 5)
    stall_count = metrics.get('stall_count', 0)

    # ABSOLUTE COMPONENT (40% weight)
    # Rewards good absolute performance regardless of path choice
    absolute_reward = (
        -0.15 * (p95_delay / 100.0)       # Absolute delay penalty
        - 0.15 * (p95_jitter / 50.0)      # Absolute jitter penalty
        + 0.10 * (throughput / 10.0)      # Absolute throughput bonus
    )

    # RELATIVE COMPONENT (60% weight)
    # Penalizes choosing worse path when better was available
    if path_metrics is not None:
        wifi = path_metrics.get('wifi', {})
        cellular = path_metrics.get('cellular', {})

        min_delay = min(wifi.get('srtt', 50), cellular.get('srtt', 80))
        min_jitter = min(wifi.get('jitter', 10), cellular.get('jitter', 15))

        # Softer normalization (75ms delay, 40ms jitter)
        relative_delay = max(0, p95_delay - min_delay)
        relative_jitter = max(0, p95_jitter - min_jitter)

        relative_reward = (
            -0.20 * (relative_delay / 75.0)    # Softer relative delay
            - 0.20 * (relative_jitter / 40.0)  # Softer relative jitter
        )
    else:
        relative_reward = 0

    # FIXED PENALTIES (unchanged)
    fixed_penalties = (
        - 0.10 * (loss_rate * 10.0)       # Loss penalty
        - 0.10 * (stall_count * 5.0)      # Stall penalty
    )

    return absolute_reward + relative_reward + fixed_penalties
```

### Reward Function Comparison

| Component | V1 (Absolute) | V2 (Relative) | V3 (Hybrid) |
|-----------|---------------|---------------|-------------|
| Delay penalty | 30% absolute | 30% relative | 15% abs + 20% rel |
| Jitter penalty | 30% absolute | 30% relative | 15% abs + 20% rel |
| Loss penalty | 20% absolute | 20% absolute | 10% absolute |
| Stall penalty | 10% absolute | 10% absolute | 10% absolute |
| Throughput bonus | 10% absolute | 10% absolute | 10% absolute |
| **Expected behavior** | WiFi bias | Exploration | Balanced |

---

## Phase 4: Comparison Framework

### Objective
Automated comparison of RL vs baselines on identical network conditions.

### Experimental Design

```
Comparison Experiment Structure:
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                         EXPERIMENT SESSION                                  │
│                                                                             │
│  Round 1: MinRTT Baseline                                                   │
│  ├── 10 episodes × 60 seconds each                                          │
│  ├── Log all metrics to baseline_minrtt_{timestamp}.csv                     │
│  └── Calculate simulated RL reward for each step                            │
│                                                                             │
│  Round 2: Weighted Round Robin Baseline                                     │
│  ├── 10 episodes × 60 seconds each                                          │
│  ├── Log all metrics to baseline_wrr_{timestamp}.csv                        │
│  └── Calculate simulated RL reward for each step                            │
│                                                                             │
│  Round 3: RL Agent (Best Model)                                             │
│  ├── 10 episodes × 60 seconds each                                          │
│  ├── Log all metrics to rl_evaluation_{timestamp}.csv                       │
│  └── Record actual RL rewards                                               │
│                                                                             │
│  Analysis:                                                                  │
│  ├── Compare mean rewards across all three                                  │
│  ├── Statistical significance test (t-test, p < 0.05)                       │
│  └── Generate comparison report                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Comparison Metrics

| Metric | Description | Success Criteria |
|--------|-------------|------------------|
| Mean Reward | Average episode reward | RL > baselines |
| Reward Variance | Consistency | RL variance ≤ baseline variance |
| P95 Delay | Latency performance | RL ≤ best baseline |
| P95 Jitter | Stability | RL ≤ best baseline |
| Throughput | Data rate | RL ≥ best baseline |
| Adaptation | Response to changes | RL adapts faster |

### Statistical Validation

```python
# Comparison analysis script (to be created)
def compare_schedulers(rl_results, minrtt_results, wrr_results):
    """
    Statistical comparison of scheduler performance
    """
    from scipy import stats

    # Paired t-test: RL vs MinRTT
    t_stat_minrtt, p_value_minrtt = stats.ttest_ind(
        rl_results['avg_reward'],
        minrtt_results['simulated_reward']
    )

    # Paired t-test: RL vs WRR
    t_stat_wrr, p_value_wrr = stats.ttest_ind(
        rl_results['avg_reward'],
        wrr_results['simulated_reward']
    )

    return {
        'rl_vs_minrtt': {'t': t_stat_minrtt, 'p': p_value_minrtt},
        'rl_vs_wrr': {'t': t_stat_wrr, 'p': p_value_wrr},
        'rl_significantly_better': p_value_minrtt < 0.05 and p_value_wrr < 0.05
    }
```

---

## Phase 5: Production Deployment Strategy

### Safety Mechanisms

#### 5.1 Fallback System

```
Production Safety Architecture:
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                         PRODUCTION SCHEDULER                                │
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                     │
│  │ RL Agent    │    │ Safety      │    │ MinRTT      │                     │
│  │ (Primary)   │───►│ Monitor     │───►│ (Fallback)  │                     │
│  └─────────────┘    └─────────────┘    └─────────────┘                     │
│         │                  │                  │                             │
│         ▼                  ▼                  ▼                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    PATH SELECTION                                    │   │
│  │                                                                      │   │
│  │  IF rl_confidence > 0.7 AND qos_ok:                                 │   │
│  │      use RL decision                                                 │   │
│  │  ELSE:                                                               │   │
│  │      use MinRTT fallback                                             │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 5.2 QoS Bounds

```kotlin
// Production safety bounds
object ProductionSafetyConfig {
    const val MAX_ACCEPTABLE_DELAY_MS = 200.0
    const val MAX_ACCEPTABLE_JITTER_MS = 50.0
    const val MAX_ACCEPTABLE_LOSS_RATE = 0.05
    const val MIN_ACCEPTABLE_THROUGHPUT_MBPS = 1.0

    const val FALLBACK_TRIGGER_CONSECUTIVE_FAILURES = 3
    const val FALLBACK_RECOVERY_WINDOW_SECONDS = 30
}
```

#### 5.3 Gradual Rollout

| Phase | RL Influence | Fallback Trigger | Duration |
|-------|--------------|------------------|----------|
| Phase A | 20% RL, 80% MinRTT | Any QoS violation | 1 week |
| Phase B | 50% RL, 50% MinRTT | 3 consecutive violations | 2 weeks |
| Phase C | 80% RL, 20% MinRTT | 5 consecutive violations | 2 weeks |
| Phase D | 100% RL | 10 consecutive violations | Ongoing |

---

## Implementation Checklist

### Phase 1: Baseline Logging (Priority: HIGH)

- [ ] Add `BaselineEpisodeLog` data class to MainActivity.kt
- [ ] Implement CSV logging for MinRTT mode
- [ ] Implement CSV logging for Round Robin mode
- [ ] Add Weighted Round Robin variant (WRR-RTT)
- [ ] Add simulated reward calculation for baselines
- [ ] Test baseline logging produces valid CSV files
- [ ] Verify metric consistency with RL logging format

### Phase 2: Extended RL Training (Priority: HIGH)

- [ ] Update MAX_EPISODES to 500
- [ ] Add convergence detection logic
- [ ] Implement early stopping if converged
- [ ] Add more detailed training progress logging
- [ ] Run 500-episode training session
- [ ] Save checkpoints at 100, 200, 300, 400, 500

### Phase 3: Reward Function (Priority: MEDIUM)

- [ ] Implement hybrid reward function (V3)
- [ ] Add reward function version selection
- [ ] Test hybrid reward on 100 episodes
- [ ] Compare V2 vs V3 performance
- [ ] Document reward function changes

### Phase 4: Comparison Framework (Priority: MEDIUM)

- [ ] Create `compare_schedulers.py` analysis script
- [ ] Implement statistical significance testing
- [ ] Create visualization tools for comparison
- [ ] Run full comparison experiment (10 episodes each)
- [ ] Generate comparison report
- [ ] Document findings

### Phase 5: Production Preparation (Priority: LOW - After Validation)

- [ ] Implement safety monitor
- [ ] Add fallback mechanism
- [ ] Create QoS bounds configuration
- [ ] Design gradual rollout plan
- [ ] Create production monitoring dashboard

---

## Success Criteria

### Minimum Requirements for Production

| Criterion | Requirement | Current Status |
|-----------|-------------|----------------|
| Training convergence | Reward variance < 0.02 over 50 episodes | ❌ Not met |
| RL vs MinRTT | RL reward > MinRTT by 20% (p < 0.05) | ❓ Not tested |
| RL vs WRR | RL reward > WRR by 15% (p < 0.05) | ❓ Not tested |
| Generalization | Performance maintained on different networks | ❓ Not tested |
| Safety bounds | QoS never drops below thresholds | ❓ Not tested |

### Decision Matrix

```
IF RL beats MinRTT by >20% AND RL beats WRR by >15%:
    → Proceed to production with safety mechanisms

IF RL beats MinRTT by 10-20% OR RL beats WRR by 10-15%:
    → Continue training, investigate reward function

IF RL performs similar to baselines (±10%):
    → RL provides no benefit, use MinRTT (simpler)

IF RL performs worse than baselines:
    → Fundamental issues, redesign approach
```

---

## Timeline Estimate

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Baseline Logging | 2-3 days | None |
| Phase 2: Extended Training | 5-7 days | Phase 1 |
| Phase 3: Reward Function | 2-3 days | Can parallel with Phase 2 |
| Phase 4: Comparison | 3-5 days | Phase 1, 2, 3 |
| Phase 5: Production Prep | 5-7 days | Phase 4 success |

**Total estimated time:** 3-4 weeks

---

## Appendix: Data Collection Requirements

### Baseline Episode CSV Schema

```csv
# File: baseline_{scheduler}_{YYYYMMDD_HHMMSS}.csv
# Schema version: 1.0

timestamp,          # ISO 8601 format
scheduler,          # "minrtt" | "round_robin" | "weighted_rr"
episode,            # 1-indexed episode number
step,               # Step within episode
wifi_srtt,          # WiFi smoothed RTT (ms)
wifi_jitter,        # WiFi jitter (ms)
wifi_loss,          # WiFi loss rate (0-1)
wifi_throughput,    # WiFi throughput (Mbps)
cell_srtt,          # Cellular smoothed RTT (ms)
cell_jitter,        # Cellular jitter (ms)
cell_loss,          # Cellular loss rate (0-1)
cell_throughput,    # Cellular throughput (Mbps)
selected_path,      # "wifi" | "cellular"
wifi_packets,       # Packets sent via WiFi this step
cell_packets,       # Packets sent via cellular this step
p95_delay,          # 95th percentile delay this step (ms)
p95_jitter,         # 95th percentile jitter this step (ms)
loss_rate,          # Loss rate this step (0-1)
throughput,         # Throughput this step (Mbps)
simulated_reward    # Reward if this were RL (for comparison)
```

### Comparison Report Template

```markdown
# Scheduler Comparison Report
Date: {date}
Network Conditions: {description}

## Summary Statistics
| Scheduler | Episodes | Mean Reward | Std Dev | P95 Delay | Throughput |
|-----------|----------|-------------|---------|-----------|------------|
| MinRTT    | N        | X.XXX       | X.XXX   | XX.X ms   | XX.X Mbps  |
| WRR       | N        | X.XXX       | X.XXX   | XX.X ms   | XX.X Mbps  |
| RL        | N        | X.XXX       | X.XXX   | XX.X ms   | XX.X Mbps  |

## Statistical Tests
- RL vs MinRTT: t=X.XX, p=X.XXXX (significant/not significant)
- RL vs WRR: t=X.XX, p=X.XXXX (significant/not significant)

## Recommendation
{Based on results, recommend: deploy RL / continue training / use baseline}
```
