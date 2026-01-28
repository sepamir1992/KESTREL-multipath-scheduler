# KESTREL RL Training System

On-device Reinforcement Learning for Multipath Packet Scheduling.

## Architecture

```
+-----------------------------------------------------------+
|         GOOGLE CLOUD VM (34.45.243.172)                   |
|                                                           |
|  Port 5000/5001: multipath_server.py (UDP packet recv)   |
|  Port 6000: rl_training_server.py (PPO training server)  |
+-----------------------------------------------------------+
                         ^
                         | TCP Socket (JSON)
                         v
+-----------------------------------------------------------+
|         ANDROID S24 ULTRA (Environment)                   |
|                                                           |
|  - Sends network state every 10 packets                   |
|  - Receives scheduling action from RL server              |
|  - Executes action and measures QoS                       |
|  - Reports reward back to server                          |
+-----------------------------------------------------------+
```

## Quick Start

### 1. Server Setup (Google Cloud VM)

```bash
# SSH to your VM
ssh user@34.45.243.172

# Clone or navigate to the project
cd ~/KESTREL-multipath-scheduler

# Install Python dependencies
pip install -r requirements.txt

# Start the multipath server (in one terminal)
python multipath_server.py

# Start the RL training server (in another terminal)
python rl_training_server.py
```

### 2. Android App Setup

1. Open the `MultipathClient` project in Android Studio
2. Sync Gradle to download Gson dependency
3. Build and install on Samsung S24 Ultra (or similar device with WiFi + Cellular)

### 3. Start Training

1. Launch the app on your Android device
2. Wait for "Both networks ready" message
3. Select **"RL Training"** radio button
4. Choose intent: `video_streaming` or `file_transfer`
5. Tap **"Start Training"**

## Files

| File | Description |
|------|-------------|
| `rl_training_server.py` | PPO agent and TCP server for RL training |
| `test_rl_connection.py` | Test script to verify server connectivity |
| `requirements.txt` | Python dependencies |
| `multipath_server.py` | Existing UDP server for packet reception |
| `MultipathClient/` | Android app with RL environment mode |

## RL Training Details

### State Space (16 dimensions)

```
WiFi metrics (7D):
  - srtt / 200.0           (Normalized RTT)
  - jitter / 50.0          (Normalized jitter)
  - burst                  (Burst indicator 0-1)
  - loss                   (Loss rate 0-1)
  - throughput / 100.0     (Normalized throughput)
  - queue_depth / 10000.0  (Normalized queue size)
  - available              (0 or 1)

Cellular metrics (7D): Same as WiFi

Intent (2D one-hot):
  - is_video_streaming
  - is_file_transfer
```

### Action Space (6 dimensions, all 0-1)

```
- weight_delay      (Weight for delay in path scoring)
- weight_jitter     (Weight for jitter)
- weight_loss       (Weight for packet loss)
- weight_throughput (Weight for throughput)
- use_wifi          (Preference for WiFi 0-1)
- use_duplication   (Enable packet duplication 0-1)
```

### Reward Functions

#### Video Streaming (V3 - Hybrid Approach)

The current reward function uses a **hybrid approach** combining absolute and relative penalties:

```python
# ABSOLUTE COMPONENT (40% weight)
# Rewards good QoS regardless of path chosen
absolute_reward = (
    -0.12 * (p95_delay / 100.0)       # Absolute delay penalty (12%)
    - 0.12 * (p95_jitter / 50.0)      # Absolute jitter penalty (12%)
    + 0.08 * (throughput / 10.0)      # Absolute throughput bonus (8%)
    + 0.08 * max(0, 1.0 - loss_rate * 20)  # Low loss bonus (8%)
)

# RELATIVE COMPONENT (60% weight)
# Penalizes choosing worse path when better available
min_delay = min(wifi_delay, cell_delay)
min_jitter = min(wifi_jitter, cell_jitter)
relative_delay = max(0, p95_delay - min_delay)
relative_jitter = max(0, p95_jitter - min_jitter)

relative_reward = (
    -0.20 * (relative_delay / 75.0)    # Relative delay penalty (20%)
    - 0.20 * (relative_jitter / 40.0)  # Relative jitter penalty (20%)
)

# FIXED PENALTIES (20% weight)
fixed_penalties = (
    - 0.10 * (loss_rate * 10.0)       # Loss penalty (10%)
    - 0.10 * (stall_count * 3.0)      # Stall penalty (10%)
)

reward = absolute_reward + relative_reward + fixed_penalties
```

**Reward Function Evolution:**

| Version | Approach | Issue |
|---------|----------|-------|
| V1 | 100% Absolute | Strong WiFi bias (86% usage) |
| V2 | 100% Relative | Poor absolute performance |
| V3 | 40% Abs + 60% Rel | Balanced exploration & performance |

#### File Transfer

```python
reward = (
    +0.6 * (throughput / 20.0)      # Maximize throughput
    -0.2 * (loss_rate * 15.0)       # Penalize loss
    +0.2 * completion_bonus         # Bonus for fast finish
)
```

## Testing

### Test RL Server Connection

```bash
# Start the RL server
python rl_training_server.py

# In another terminal, run the test
python test_rl_connection.py --host localhost --port 6000 --episodes 5
```

### Test with Remote Server

```bash
python test_rl_connection.py --host 34.45.243.172 --port 6000 --episodes 10
```

## Model Outputs

The training server saves models to the `models/` directory:

| File | Description |
|------|-------------|
| `kestrel_model_BEST.pth` | Best model (highest avg reward) |
| `kestrel_model_episode_N.pth` | Checkpoint every 10 episodes |
| `kestrel_model_FINAL.pth` | Model saved on server shutdown |
| `training_log.json` | Training statistics per episode |

## CSV Training Data

The Android app logs training data to:
```
/sdcard/Android/data/com.example.multipathclient/files/kestrel_training_YYYYMMDD_HHMMSS.csv
```

Columns:
- timestamp, episode, step, intent
- wifi_srtt, wifi_jitter, wifi_burst, wifi_loss, wifi_throughput, wifi_queue, wifi_available
- cell_srtt, cell_jitter, cell_burst, cell_loss, cell_throughput, cell_queue, cell_available
- action_w_delay, action_w_jitter, action_w_loss, action_w_throughput, action_use_wifi, action_use_duplication
- wifi_packets, cell_packets, p95_delay, p95_jitter, loss_rate, throughput, reward, cumulative_reward

## PPO Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 3e-4 |
| Gamma (discount) | 0.99 |
| GAE Lambda | 0.95 |
| PPO Clip Epsilon | 0.2 |
| PPO Epochs | 10 |
| Entropy Coefficient | 0.01 |
| Value Loss Coefficient | 0.5 |
| Max Gradient Norm | 0.5 |

## Episode Structure

- **Episode duration:** 60 seconds
- **Step:** 10 packets (~1-2 seconds)
- **Steps per episode:** ~30-60

## Network Requirements

### Real Network Interface Architecture

The Android client uses **real WiFi and cellular hardware interfaces**, not simulation:

```
┌─────────────────────────────────────────────────────────────┐
│                      Android Phone                          │
│                                                             │
│  ┌──────────────────┐          ┌──────────────────┐        │
│  │   WiFi Radio     │          │  5G/LTE Radio    │        │
│  │   (Real HW)      │          │   (Real HW)      │        │
│  └────────┬─────────┘          └────────┬─────────┘        │
│           │                             │                   │
│           ▼                             ▼                   │
│  ┌──────────────────┐          ┌──────────────────┐        │
│  │   wifiSocket     │          │  cellularSocket  │        │
│  │  (UDP, bound to  │          │  (UDP, bound to  │        │
│  │   WiFi NIC)      │          │   Cellular NIC)  │        │
│  └────────┬─────────┘          └────────┬─────────┘        │
│           │                             │                   │
└───────────┼─────────────────────────────┼───────────────────┘
            │                             │
            ▼                             ▼
     ┌────────────┐                ┌────────────┐
     │  WiFi AP   │                │ Cell Tower │
     └──────┬─────┘                └──────┬─────┘
            │                             │
            └───────────┬─────────────────┘
                        ▼
              ┌───────────────────┐
              │   Server (VM)     │
              │  Port 5000 (WiFi) │
              │  Port 5001 (Cell) │
              │  Port 6000 (RL)   │
              └───────────────────┘
```

**Key Points:**
- Each socket is bound to a specific network interface using Android's `Network.bindSocket()` API
- Packets sent via `wifiSocket` traverse the actual WiFi radio
- Packets sent via `cellularSocket` traverse the actual 5G/LTE radio
- All metrics (RTT, jitter, loss) are real measurements from actual packet transmissions

### Server Firewall Rules

Ensure your Google Cloud VM allows inbound traffic on:
- **TCP 6000** - RL training server
- **UDP 5000** - WiFi path packets
- **UDP 5001** - Cellular path packets

### Android Permissions

The app requires:
- `INTERNET`
- `ACCESS_NETWORK_STATE`
- `CHANGE_NETWORK_STATE`
- `WAKE_LOCK`

## Troubleshooting

### "Cannot connect to RL server"

1. Check firewall rules on Google Cloud
2. Verify server IP address in `MainActivity.kt`
3. Ensure `rl_training_server.py` is running

### "Both networks must be available"

1. Connect to WiFi
2. Enable mobile data
3. Disable "WiFi-only" mode in Android settings

### Training is slow

- Reduce `EPISODE_DURATION_MS` for faster iterations
- Reduce `PACKETS_PER_STEP` for more frequent actions
- Check network latency to server

## Expected Training Behavior

After ~50-100 episodes:

**Video Streaming Intent:**
- Higher `weight_delay` and `weight_jitter`
- Balanced path usage (not extreme WiFi bias with V3 reward)

**File Transfer Intent:**
- Higher `weight_throughput`
- Path selection based on current throughput

## Baseline Comparison

### Available Schedulers

| Scheduler | Description | Use Case |
|-----------|-------------|----------|
| **MinRTT** | Always choose path with lowest RTT | Simple baseline |
| **Round Robin** | Alternate between WiFi and Cellular | Load balancing baseline |
| **Weighted RR** | Probabilistic selection based on inverse RTT | Adaptive baseline |
| **RL (KESTREL)** | Learned policy from PPO training | Experimental |

### Running Comparisons

```bash
# After training, analyze results
python compare_schedulers.py --rl-log models/training_log.json

# Generate comparison report
python compare_schedulers.py --rl-log models/training_log.json --output report.txt
```

### Comparison Metrics

The comparison tool evaluates:

| Metric | Description | RL Must Beat Baseline By |
|--------|-------------|--------------------------|
| Mean Reward | Average episode reward | >20% for deployment |
| P95 Delay | 95th percentile latency | Must be equal or better |
| Throughput | Data rate achieved | Must be equal or better |
| Stability | Reward variance | Must be equal or better |

### Validation Criteria

```
IF RL beats MinRTT by >20% AND beats WRR by >15%:
    → Ready for production deployment

IF RL beats baselines by 10-20%:
    → Continue training, need more episodes

IF RL performs within ±10% of baselines:
    → Use MinRTT (simpler, no training needed)

IF RL performs worse than baselines:
    → Review reward function and state representation
```

## Training Configuration

### Current Settings

| Parameter | Value | Notes |
|-----------|-------|-------|
| Max Episodes | 500 | Increased for convergence |
| Checkpoint Interval | 10 episodes | Regular model saves |
| Episode Duration | 60 seconds | Per-episode time |
| Packets per Step | 10 | Action frequency |

### Convergence Monitoring

Training is considered converged when:
- Rolling 50-episode average changes < 2%
- Reward variance decreases over time
- WiFi usage stabilizes (not oscillating wildly)

## Documentation

Additional documentation available in `docs/`:

| Document | Description |
|----------|-------------|
| `ROADMAP_RL_VALIDATION.md` | Complete validation roadmap and next steps |
| `training_analysis_video_streaming_v1.md` | Analysis of V1 (absolute) training run |
| `training_analysis_video_streaming_v2_new_reward.md` | Analysis of V2 (relative) training run |

## References

- PPO Paper: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
- KESTREL: On-Device RL for Multipath Scheduling
