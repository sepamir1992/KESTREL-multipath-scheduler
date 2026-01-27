# Video Streaming Training Analysis - Run 1

**Date:** January 2026
**Episodes:** 53
**Intent:** video_streaming
**Reward Function:** Absolute penalties (pre-modification)

## Executive Summary

This document analyzes the first training run of the KESTREL RL-based multipath scheduler for video streaming intent. The agent successfully learned to prefer WiFi (the lower-jitter path) but showed signs of learning a static policy rather than adapting to network conditions.

**Key Finding:** The reward function's absolute penalties may have been too harsh on cellular usage, causing the agent to learn "always use WiFi" rather than "choose the path with better metrics."

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO (Proximal Policy Optimization) |
| Learning Rate | 3e-4 |
| Discount Factor (γ) | 0.99 |
| GAE Lambda | 0.95 |
| PPO Clip Epsilon | 0.2 |
| Entropy Coefficient | 0.01 |
| Episode Duration | ~60 seconds |
| Steps per Episode | 60-82 |

### Reward Function (Absolute Penalties)

```python
reward = (
    -0.3 * (p95_delay / 100.0)      # 30% weight on delay
    -0.3 * (p95_jitter / 50.0)      # 30% weight on jitter
    -0.2 * (loss_rate * 10.0)       # 20% weight on loss
    -0.1 * (stall_count * 5.0)      # 10% weight on stalls
    +0.1 * (throughput / 5.0)       # 10% weight on throughput
)
```

---

## Overall Performance Metrics

| Metric | Value |
|--------|-------|
| Total Episodes | 53 |
| Best Episode | #14 (avg_reward: -0.261) |
| Worst Episode | #19 (avg_reward: -0.487) |
| Mean Avg Reward | -0.327 |
| Std Dev Avg Reward | 0.058 |
| Mean WiFi Usage | 86.3% |
| Mean Steps/Episode | 69.4 |

---

## Learning Progression

### Reward Trend by Episode Window

| Episode Range | Mean Avg Reward | Trend |
|---------------|-----------------|-------|
| 1-10 | -0.338 | Baseline |
| 11-20 | -0.371 | Exploration dip |
| 21-30 | -0.344 | Recovery |
| 31-40 | -0.314 | Improvement |
| 41-53 | -0.314 | Stabilization |

### Visualization (ASCII)

```
Avg Reward per Episode:
-0.26 |          *
-0.28 |    *        *
-0.30 |  * * *   **  * *    *  * ** ****  *
-0.32 | *     *       *  ****  *      * ** **
-0.35 |*       *        *                   *
-0.38 |                *
-0.40 |                 *
-0.44 |               *
-0.49 |              *
      +--------------------------------------------
       1    10   20   30   40   50   Episode
```

---

## WiFi vs Cellular Usage Analysis

### Usage Distribution

| WiFi Usage Range | Episodes | Mean Reward | Observation |
|------------------|----------|-------------|-------------|
| > 95% | 7 | -0.295 | Best performance |
| 85-95% | 19 | -0.322 | Good performance |
| 75-85% | 15 | -0.338 | Moderate performance |
| < 75% | 12 | -0.392 | Poor performance |

### Correlation Analysis

**WiFi Usage vs Reward:** Strong positive correlation (r ≈ 0.65)

Higher WiFi usage consistently correlated with better rewards, indicating:
1. WiFi had better network metrics (lower jitter/delay) in the test environment
2. The agent learned this relationship correctly
3. However, this may lead to over-reliance on WiFi

### WiFi Usage Trend

```
WiFi Usage % per Episode:
100% |    *  * *    *  **   *    ** *  *  *
 95% |  ** ** * ********  ****  *   * **  ** *
 90% |**         *      **    **     *
 85% |              *       *    *   *  ** **
 80% |         *              *  **    *
 75% |*
 70% |              *
 65% |               **
 60% |                *
     +--------------------------------------------
      1    10   20   30   40   50   Episode
```

---

## Best Episodes Analysis

| Rank | Episode | Avg Reward | WiFi % | Steps | Key Characteristics |
|------|---------|------------|--------|-------|---------------------|
| 1 | 14 | -0.261 | 98.7% | 82 | Highest WiFi, most steps |
| 2 | 12 | -0.272 | 93.0% | 77 | High WiFi, many steps |
| 3 | 6 | -0.281 | 95.3% | 76 | Consistent WiFi preference |
| 4 | 5 | -0.291 | 90.0% | 69 | Early learning success |
| 5 | 26 | -0.274 | 93.2% | 69 | Mid-training optimization |

**Common Pattern:** Best episodes all have:
- WiFi usage > 90%
- Above-average step count (indicating smooth packet delivery)
- No major exploration of cellular path

---

## Worst Episodes Analysis

| Rank | Episode | Avg Reward | WiFi % | Steps | Likely Cause |
|------|---------|------------|--------|-------|--------------|
| 1 | 19 | -0.487 | 70.5% | 61 | Cellular exploration |
| 2 | 21 | -0.485 | 66.6% | 61 | Continued exploration |
| 3 | 23 | -0.480 | 97.9% | 63 | Network degradation? |
| 4 | 22 | -0.450 | 60.5% | 60 | Heavy cellular use |
| 5 | 11 | -0.443 | 73.7% | 68 | Exploration phase |

**Observations:**
- Episodes 19-22 form a cluster of poor performance
- This appears to be PPO exploring alternative strategies
- The agent learned these strategies don't work and returned to WiFi preference
- Episode 23 is unusual: high WiFi but poor reward (possible network issue)

---

## Exploration vs Exploitation

### Exploration Dip (Episodes 18-23)

A significant performance drop occurred during episodes 18-23:

```
Episode 18: reward=-0.438, WiFi=87.6%
Episode 19: reward=-0.487, WiFi=70.5%  <- Worst
Episode 20: reward=-0.399, WiFi=65.6%
Episode 21: reward=-0.485, WiFi=66.6%
Episode 22: reward=-0.450, WiFi=60.5%  <- Lowest WiFi
Episode 23: reward=-0.480, WiFi=97.9%
```

**Interpretation:**
1. The PPO entropy bonus encouraged exploration of cellular-heavy strategies
2. The agent tested using more cellular (down to 60.5% WiFi)
3. Rewards dropped significantly, teaching the agent cellular is suboptimal
4. By episode 24, the agent returned to WiFi-dominant strategy

This is **healthy exploration** - the agent needed to confirm WiFi is better.

---

## Steps per Episode Analysis

Steps per episode indicates packet throughput efficiency:

| Steps Range | Episodes | Mean Reward | Interpretation |
|-------------|----------|-------------|----------------|
| 75+ | 10 | -0.294 | Excellent throughput |
| 70-74 | 12 | -0.314 | Good throughput |
| 65-69 | 18 | -0.333 | Average throughput |
| < 65 | 13 | -0.379 | Below average |

**Correlation:** More steps = better reward (r ≈ 0.45)

Higher step counts indicate:
- Faster packet delivery
- Less time waiting for retransmissions
- Better overall network conditions

---

## Convergence Analysis

### Has the Agent Converged?

**Partial convergence observed:**

1. **Policy Behavior:** Stabilized around 85-95% WiFi preference
2. **Reward Variance:** Still moderate (σ = 0.058)
3. **Exploration:** Occasional dips suggest continued exploration

### Convergence Indicators

| Indicator | Status | Evidence |
|-----------|--------|----------|
| Reward plateau | Partial | Episodes 31-53 relatively stable |
| Policy stability | Yes | Consistent WiFi preference |
| Variance reduction | Partial | Still seeing ±0.1 swings |
| Exploration decay | No | Entropy still causing exploration |

**Recommendation:** Continue training to 100+ episodes for full convergence.

---

## Identified Issues

### Issue 1: Absolute Penalty Bias

The reward function penalizes delay/jitter absolutely, not relative to alternatives:

```python
# Current: Always penalizes high jitter
-0.3 * (p95_jitter / 50.0)

# Problem: If WiFi=20ms and Cell=40ms jitter
# Agent learns "WiFi good" not "lower jitter good"
```

**Impact:** Agent may not adapt if cellular becomes better than WiFi.

### Issue 2: Limited Generalization

The agent learned a policy specific to this network environment:
- WiFi consistently had lower jitter
- Agent learned "prefer WiFi" as a shortcut
- May not generalize to environments where cellular is better

### Issue 3: Insufficient Episodes

53 episodes may not be enough for:
- Full policy convergence
- Robust generalization
- Confidence in learned behavior

---

## Recommendations

### Immediate Actions

1. **Extend training to 100 episodes** - Allow more time for convergence
2. **Implement relative penalties** - Penalize based on deviation from best available path
3. **Monitor per-path metrics** - Track when cellular outperforms WiFi

### Reward Function Modification (Implemented)

```python
# New: Relative penalties
min_delay = min(wifi_delay, cell_delay)
min_jitter = min(wifi_jitter, cell_jitter)

relative_delay = max(0, p95_delay - min_delay)
relative_jitter = max(0, p95_jitter - min_jitter)

reward = (
    -0.3 * (relative_delay / 50.0)    # Only penalize if worse than best
    -0.3 * (relative_jitter / 25.0)   # Only penalize if worse than best
    ...
)
```

### Future Improvements

1. **Curriculum learning** - Start with simple scenarios, increase complexity
2. **Domain randomization** - Vary network conditions to improve generalization
3. **Multi-environment testing** - Validate on different network setups

---

## Raw Data Summary

### Episode Statistics

| Statistic | Avg Reward | WiFi % | Steps | Duration (s) |
|-----------|------------|--------|-------|--------------|
| Mean | -0.327 | 86.3 | 69.4 | 60.5 |
| Std Dev | 0.058 | 10.2 | 5.4 | 0.4 |
| Min | -0.487 | 60.5 | 60 | 60.1 |
| Max | -0.261 | 99.9 | 82 | 62.3 |
| Median | -0.314 | 88.9 | 69 | 60.5 |

### Best Model Checkpoints

| Episode | Avg Reward | Status |
|---------|------------|--------|
| 14 | -0.261 | BEST |
| 12 | -0.272 | Previous best |
| 6 | -0.281 | Previous best |
| 5 | -0.291 | Previous best |

---

## Conclusion

The first training run demonstrates that the KESTREL RL agent can successfully learn to optimize video streaming by preferring lower-jitter paths. However, the absolute penalty structure may have created an overly rigid policy that prefers WiFi regardless of actual network conditions.

The implemented changes (relative penalties + 100 episodes) should address these concerns and produce a more adaptive agent that responds to real-time network metrics rather than learned path preferences.

**Next Steps:**
1. Run training with modified reward function
2. Compare learning curves and final performance
3. Test with scenarios where cellular outperforms WiFi
