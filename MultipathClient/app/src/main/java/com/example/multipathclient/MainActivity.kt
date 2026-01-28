package com.example.multipathclient

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.net.*
import android.os.Bundle
import android.os.Environment
import android.os.PowerManager
import android.util.Log
import android.view.View
import android.widget.*
import android.widget.ScrollView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.gson.Gson
import com.google.gson.annotations.SerializedName
import java.io.*
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetAddress
import java.net.Socket
import java.nio.ByteBuffer
import java.text.SimpleDateFormat
import java.util.*
import kotlin.concurrent.thread
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

class MainActivity : AppCompatActivity() {
    private lateinit var statusText: TextView
    private lateinit var startButton: Button
    private lateinit var testTypeRadioGroup: RadioGroup
    private lateinit var intentSpinner: Spinner
    private lateinit var intentSpinnerLayout: LinearLayout
    private lateinit var rlStatusText: TextView
    private lateinit var rlStatusScrollView: ScrollView
    private lateinit var connectivityManager: ConnectivityManager

    private var wifiNetwork: Network? = null
    private var cellularNetwork: Network? = null
    private var wifiSocket: DatagramSocket? = null
    private var cellularSocket: DatagramSocket? = null
    private var rlSocket: Socket? = null
    private var rlReader: BufferedReader? = null
    private var rlWriter: PrintWriter? = null

    // RTT and telemetry
    private var wifiRTT = 100.0
    private var cellularRTT = 100.0
    private var wifiJitter = 10.0
    private var cellularJitter = 10.0
    private var wifiLoss = 0.0
    private var cellularLoss = 0.0
    private var wifiThroughput = 0.0
    private var cellularThroughput = 0.0
    private var wifiQueueDepth = 0
    private var cellularQueueDepth = 0
    private var wifiBurst = 0.0
    private var cellularBurst = 0.0

    // RTT history for burst calculation
    private val wifiRttHistory = mutableListOf<Double>()
    private val cellularRttHistory = mutableListOf<Double>()
    private val RTT_HISTORY_SIZE = 20

    // Packet counting
    private var wifiPacketsSent = 0
    private var wifiPacketsAcked = 0
    private var cellularPacketsSent = 0
    private var cellularPacketsAcked = 0
    private var wifiBytesSent = 0L
    private var cellularBytesSent = 0L

    // Step-level tracking
    private var stepWifiPackets = 0
    private var stepCellPackets = 0
    private val stepRttSamples = mutableListOf<Double>()
    private val stepJitterSamples = mutableListOf<Double>()
    private var stepBytesSent = 0L
    private var stepStartTime = 0L
    private var stepLostPackets = 0
    private var stepTotalPackets = 0

    // RL Training state
    private var currentEpisode = 0
    private var currentStep = 0
    private var episodeReward = 0.0
    private var bestEpisodeReward = Double.NEGATIVE_INFINITY
    private var currentAction: SchedulingAction? = null
    private var selectedIntent = "video_streaming"

    private var isRunning = false
    private val gson = Gson()

    private enum class TestType { MIN_RTT, ROUND_ROBIN, WEIGHTED_RR, RL_TRAINING }
    private var selectedTestType = TestType.MIN_RTT

    // Baseline logging
    private var baselineCsvWriter: BufferedWriter? = null
    private var baselineEpisode = 0
    private var baselineStep = 0
    private var baselineStartTime = 0L

    private val SERVER_IP = "34.45.243.172"
    private val WIFI_PORT = 5000
    private val CELLULAR_PORT = 5001
    private val RL_PORT = 6000
    private val CHUNK_SIZE = 1400
    private val PACKETS_PER_STEP = 10
    private val EPISODE_DURATION_MS = 60_000L

    // CSV Logging
    private var csvWriter: BufferedWriter? = null
    private var csvFilePath: String? = null

    // ============================================================
    // Data Classes for JSON Communication
    // ============================================================

    data class WiFiMetrics(
        val srtt: Double,
        val jitter: Double,
        val burst: Double,
        val loss: Double,
        val throughput: Double,
        @SerializedName("queue_depth") val queueDepth: Int,
        val available: Boolean
    )

    data class CellularMetrics(
        val srtt: Double,
        val jitter: Double,
        val burst: Double,
        val loss: Double,
        val throughput: Double,
        @SerializedName("queue_depth") val queueDepth: Int,
        val available: Boolean
    )

    data class NetworkState(
        val wifi: WiFiMetrics,
        val cellular: CellularMetrics,
        val intent: String
    )

    data class GetActionRequest(
        val type: String = "get_action",
        val state: NetworkState
    )

    data class SchedulingAction(
        @SerializedName("weight_delay") val weightDelay: Double,
        @SerializedName("weight_jitter") val weightJitter: Double,
        @SerializedName("weight_loss") val weightLoss: Double,
        @SerializedName("weight_throughput") val weightThroughput: Double,
        @SerializedName("use_wifi") val useWifi: Double,
        @SerializedName("use_duplication") val useDuplication: Double
    )

    data class ActionResponse(
        val action: SchedulingAction
    )

    data class StepMetrics(
        @SerializedName("p95_delay") val p95Delay: Double,
        @SerializedName("p95_jitter") val p95Jitter: Double,
        @SerializedName("loss_rate") val lossRate: Double,
        val throughput: Double,
        @SerializedName("stall_count") val stallCount: Int,
        @SerializedName("bytes_sent") val bytesSent: Long,
        @SerializedName("completion_time") val completionTime: Double,
        @SerializedName("wifi_packets") val wifiPackets: Int,
        @SerializedName("cell_packets") val cellPackets: Int
    )

    data class ReportRewardRequest(
        val type: String = "report_reward",
        val metrics: StepMetrics,
        val intent: String,
        val done: Boolean
    )

    data class RewardResponse(
        val status: String,
        val reward: Double,
        @SerializedName("episode_reward") val episodeReward: Double
    )

    data class EpisodeDoneRequest(
        val type: String = "episode_done"
    )

    data class EpisodeDoneResponse(
        val status: String,
        val episode: Int,
        @SerializedName("total_reward") val totalReward: Double,
        @SerializedName("avg_reward") val avgReward: Double,
        @SerializedName("is_best") val isBest: Boolean
    )

    // ============================================================
    // Activity Lifecycle
    // ============================================================

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        statusText = findViewById(R.id.statusText)
        startButton = findViewById(R.id.startButton)
        testTypeRadioGroup = findViewById(R.id.testTypeRadioGroup)
        intentSpinner = findViewById(R.id.intentSpinner)
        intentSpinnerLayout = findViewById(R.id.intentSpinnerLayout)
        rlStatusText = findViewById(R.id.rlStatusText)
        rlStatusScrollView = findViewById(R.id.rlStatusScrollView)

        startButton.isEnabled = false

        connectivityManager = getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager

        // Request permissions
        val permissions = arrayOf(
            Manifest.permission.ACCESS_NETWORK_STATE,
            Manifest.permission.CHANGE_NETWORK_STATE,
            Manifest.permission.INTERNET,
            Manifest.permission.WAKE_LOCK
        )
        if (permissions.any { ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED }) {
            ActivityCompat.requestPermissions(this, permissions, 100)
        }

        // Setup intent spinner
        val intents = arrayOf("video_streaming", "file_transfer")
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, intents)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        intentSpinner.adapter = adapter
        intentSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                selectedIntent = intents[position]
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }

        // Setup test type radio group
        testTypeRadioGroup.setOnCheckedChangeListener { _, checkedId ->
            selectedTestType = when (checkedId) {
                R.id.roundRobinRadioButton -> TestType.ROUND_ROBIN
                R.id.weightedRRRadioButton -> TestType.WEIGHTED_RR
                R.id.rlTrainingRadioButton -> TestType.RL_TRAINING
                else -> TestType.MIN_RTT
            }
            // Show/hide RL-specific UI
            val isRlMode = selectedTestType == TestType.RL_TRAINING
            intentSpinnerLayout.visibility = if (isRlMode) View.VISIBLE else View.GONE
            rlStatusScrollView.visibility = if (isRlMode) View.VISIBLE else View.GONE
        }

        startButton.setOnClickListener {
            if (!isRunning) {
                if (selectedTestType == TestType.RL_TRAINING) {
                    startRLTraining()
                } else {
                    startMultipathTest()
                }
            } else {
                stopMultipathTest()
            }
        }

        setupNetworks()
    }

    override fun onDestroy() {
        super.onDestroy()
        stopMultipathTest()
        csvWriter?.close()
    }

    // ============================================================
    // Network Setup
    // ============================================================

    private fun checkNetworksAndEnableButton() {
        if (wifiNetwork != null && cellularNetwork != null) {
            runOnUiThread {
                startButton.isEnabled = true
                updateStatus("Both networks ready. Press Start.")
            }
        }
    }

    private fun setupNetworks() {
        updateStatus("Setting up networks...")
        runOnUiThread { startButton.isEnabled = false }

        val wifiRequest = NetworkRequest.Builder()
            .addTransportType(NetworkCapabilities.TRANSPORT_WIFI)
            .build()

        connectivityManager.requestNetwork(wifiRequest, object : ConnectivityManager.NetworkCallback() {
            override fun onAvailable(network: Network) {
                wifiNetwork = network
                updateNetworkStatus()
                checkNetworksAndEnableButton()
            }

            override fun onLost(network: Network) {
                wifiNetwork = null
                runOnUiThread { startButton.isEnabled = false }
                updateNetworkStatus()
            }
        })

        val cellularRequest = NetworkRequest.Builder()
            .addTransportType(NetworkCapabilities.TRANSPORT_CELLULAR)
            .build()

        connectivityManager.requestNetwork(cellularRequest, object : ConnectivityManager.NetworkCallback() {
            override fun onAvailable(network: Network) {
                cellularNetwork = network
                updateNetworkStatus()
                checkNetworksAndEnableButton()
            }

            override fun onLost(network: Network) {
                cellularNetwork = null
                runOnUiThread { startButton.isEnabled = false }
                updateNetworkStatus()
            }
        })
    }

    // ============================================================
    // Telemetry Functions
    // ============================================================

    private fun updateRttHistory(rtt: Double, isWifi: Boolean) {
        val history = if (isWifi) wifiRttHistory else cellularRttHistory
        history.add(rtt)
        if (history.size > RTT_HISTORY_SIZE) {
            history.removeAt(0)
        }
    }

    private fun calculateBurst(isWifi: Boolean): Double {
        val history = if (isWifi) wifiRttHistory else cellularRttHistory
        val srtt = if (isWifi) wifiRTT else cellularRTT
        if (history.size < 2) return 0.0

        val maxRtt = history.maxOrNull() ?: 0.0
        val minRtt = history.minOrNull() ?: 0.0
        return (maxRtt - minRtt) / (srtt + 5.0)
    }

    private fun calculateP95(samples: List<Double>): Double {
        if (samples.isEmpty()) return 0.0
        val sorted = samples.sorted()
        val index = (sorted.size * 0.95).toInt().coerceIn(0, sorted.size - 1)
        return sorted[index]
    }

    private fun updateJitter(newRtt: Double, isWifi: Boolean) {
        val prevRtt = if (isWifi) wifiRTT else cellularRTT
        val jitterSample = abs(newRtt - prevRtt)
        if (isWifi) {
            wifiJitter = 0.875 * wifiJitter + 0.125 * jitterSample
        } else {
            cellularJitter = 0.875 * cellularJitter + 0.125 * jitterSample
        }
        stepJitterSamples.add(jitterSample)
    }

    private fun updateLossRate(acked: Boolean, isWifi: Boolean) {
        if (isWifi) {
            wifiPacketsSent++
            if (acked) wifiPacketsAcked++
            wifiLoss = 1.0 - (wifiPacketsAcked.toDouble() / max(wifiPacketsSent, 1))
        } else {
            cellularPacketsSent++
            if (acked) cellularPacketsAcked++
            cellularLoss = 1.0 - (cellularPacketsAcked.toDouble() / max(cellularPacketsSent, 1))
        }
    }

    private fun updateThroughput(bytes: Int, timeMs: Long, isWifi: Boolean) {
        if (timeMs <= 0) return
        val mbps = (bytes * 8.0) / (timeMs * 1000.0) // Mbps
        if (isWifi) {
            wifiThroughput = 0.875 * wifiThroughput + 0.125 * mbps
        } else {
            cellularThroughput = 0.875 * cellularThroughput + 0.125 * mbps
        }
    }

    // ============================================================
    // RL Communication
    // ============================================================

    private fun connectToRLServer(): Boolean {
        return try {
            rlSocket = Socket(SERVER_IP, RL_PORT)
            rlSocket?.soTimeout = 5000
            rlReader = BufferedReader(InputStreamReader(rlSocket!!.getInputStream()))
            rlWriter = PrintWriter(BufferedWriter(OutputStreamWriter(rlSocket!!.getOutputStream())), true)
            Log.d("RL", "Connected to RL server at $SERVER_IP:$RL_PORT")
            true
        } catch (e: Exception) {
            Log.e("RL", "Failed to connect to RL server: ${e.message}")
            false
        }
    }

    private fun disconnectFromRLServer() {
        try {
            rlReader?.close()
            rlWriter?.close()
            rlSocket?.close()
        } catch (e: Exception) {
            Log.e("RL", "Error closing RL connection: ${e.message}")
        }
        rlReader = null
        rlWriter = null
        rlSocket = null
    }

    private fun sendRLMessage(message: String): String? {
        return try {
            rlWriter?.println(message)
            rlReader?.readLine()
        } catch (e: Exception) {
            Log.e("RL", "RL communication error: ${e.message}")
            null
        }
    }

    private fun getActionFromRL(): SchedulingAction? {
        val state = NetworkState(
            wifi = WiFiMetrics(
                srtt = wifiRTT,
                jitter = wifiJitter,
                burst = wifiBurst,
                loss = wifiLoss,
                throughput = wifiThroughput,
                queueDepth = wifiQueueDepth,
                available = wifiNetwork != null
            ),
            cellular = CellularMetrics(
                srtt = cellularRTT,
                jitter = cellularJitter,
                burst = cellularBurst,
                loss = cellularLoss,
                throughput = cellularThroughput,
                queueDepth = cellularQueueDepth,
                available = cellularNetwork != null
            ),
            intent = selectedIntent
        )

        val request = GetActionRequest(state = state)
        val requestJson = gson.toJson(request)
        val responseJson = sendRLMessage(requestJson)

        return if (responseJson != null) {
            try {
                val response = gson.fromJson(responseJson, ActionResponse::class.java)
                response.action
            } catch (e: Exception) {
                Log.e("RL", "Failed to parse action response: ${e.message}")
                null
            }
        } else null
    }

    private fun reportRewardToRL(metrics: StepMetrics, done: Boolean): Double {
        val request = ReportRewardRequest(
            metrics = metrics,
            intent = selectedIntent,
            done = done
        )
        val requestJson = gson.toJson(request)
        val responseJson = sendRLMessage(requestJson)

        return if (responseJson != null) {
            try {
                val response = gson.fromJson(responseJson, RewardResponse::class.java)
                response.reward
            } catch (e: Exception) {
                Log.e("RL", "Failed to parse reward response: ${e.message}")
                0.0
            }
        } else 0.0
    }

    private fun notifyEpisodeDone(): EpisodeDoneResponse? {
        val request = EpisodeDoneRequest()
        val requestJson = gson.toJson(request)
        val responseJson = sendRLMessage(requestJson)

        return if (responseJson != null) {
            try {
                gson.fromJson(responseJson, EpisodeDoneResponse::class.java)
            } catch (e: Exception) {
                Log.e("RL", "Failed to parse episode done response: ${e.message}")
                null
            }
        } else null
    }

    // ============================================================
    // Path Scoring (for RL mode)
    // ============================================================

    private fun calculatePathScore(
        srtt: Double,
        jitter: Double,
        loss: Double,
        throughput: Double,
        action: SchedulingAction
    ): Double {
        // Normalize metrics to 0-1
        val normDelay = srtt / 200.0
        val normJitter = jitter / 50.0
        val normLoss = loss
        val normThroughput = 1.0 - (throughput / 100.0)

        // Weighted score (lower = better)
        return (
            action.weightDelay * normDelay +
            action.weightJitter * normJitter +
            action.weightLoss * normLoss +
            action.weightThroughput * normThroughput
        )
    }

    private fun selectPathUsingRL(action: SchedulingAction): Boolean {
        val wifiScore = calculatePathScore(wifiRTT, wifiJitter, wifiLoss, wifiThroughput, action)
        val cellScore = calculatePathScore(cellularRTT, cellularJitter, cellularLoss, cellularThroughput, action)

        return if (abs(wifiScore - cellScore) < 0.1) {
            // Close scores - use action's preference
            Math.random() < action.useWifi
        } else {
            wifiScore < cellScore
        }
    }

    /**
     * Weighted Round Robin path selection.
     * Probabilistically selects path based on inverse RTT.
     * Lower RTT = higher probability of selection.
     */
    private fun selectPathWeightedRR(): Boolean {
        // Weight inversely proportional to RTT (lower RTT = higher weight)
        val wifiWeight = 1.0 / (wifiRTT + 1.0)
        val cellWeight = 1.0 / (cellularRTT + 1.0)
        val totalWeight = wifiWeight + cellWeight
        val wifiProbability = wifiWeight / totalWeight
        return Math.random() < wifiProbability
    }

    // ============================================================
    // Baseline CSV Logging (for MinRTT, RR, Weighted RR)
    // ============================================================

    private fun initializeBaselineCsvLogging() {
        try {
            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val schedulerName = selectedTestType.name.lowercase()
            val filename = "baseline_${schedulerName}_${timestamp}.csv"
            val dir = getExternalFilesDir(null)
            val file = File(dir, filename)

            baselineCsvWriter = BufferedWriter(FileWriter(file))
            baselineCsvWriter?.write(
                "timestamp,scheduler,episode,step," +
                "wifi_srtt,wifi_jitter,wifi_loss,wifi_throughput," +
                "cell_srtt,cell_jitter,cell_loss,cell_throughput," +
                "selected_path,wifi_packets,cell_packets," +
                "p95_delay,p95_jitter,loss_rate,throughput,simulated_reward\n"
            )

            baselineEpisode = 1
            baselineStep = 0
            baselineStartTime = System.currentTimeMillis()

            Log.d("Baseline", "CSV logging initialized: ${file.absolutePath}")
        } catch (e: Exception) {
            Log.e("Baseline", "Failed to initialize CSV logging: ${e.message}")
        }
    }

    private fun logBaselineStep(
        selectedPath: String,
        wifiPackets: Int,
        cellPackets: Int,
        p95Delay: Double,
        p95Jitter: Double,
        lossRate: Double,
        throughput: Double
    ) {
        try {
            baselineStep++
            val timestamp = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS", Locale.US).format(Date())
            val schedulerName = selectedTestType.name.lowercase()

            // Calculate simulated reward (what RL would have gotten)
            val simulatedReward = calculateSimulatedReward(
                p95Delay, p95Jitter, lossRate, throughput,
                wifiRTT, wifiJitter, cellularRTT, cellularJitter
            )

            val line = "$timestamp,$schedulerName,$baselineEpisode,$baselineStep," +
                "${"%.2f".format(wifiRTT)},${"%.2f".format(wifiJitter)},${"%.4f".format(wifiLoss)},${"%.2f".format(wifiThroughput)}," +
                "${"%.2f".format(cellularRTT)},${"%.2f".format(cellularJitter)},${"%.4f".format(cellularLoss)},${"%.2f".format(cellularThroughput)}," +
                "$selectedPath,$wifiPackets,$cellPackets," +
                "${"%.2f".format(p95Delay)},${"%.2f".format(p95Jitter)},${"%.4f".format(lossRate)},${"%.2f".format(throughput)},${"%.4f".format(simulatedReward)}\n"

            baselineCsvWriter?.write(line)
            baselineCsvWriter?.flush()
        } catch (e: Exception) {
            Log.e("Baseline", "Failed to log step: ${e.message}")
        }
    }

    /**
     * Calculate what reward the RL agent would have received for these metrics.
     * Uses V3 hybrid reward function for fair comparison.
     */
    private fun calculateSimulatedReward(
        p95Delay: Double,
        p95Jitter: Double,
        lossRate: Double,
        throughput: Double,
        wifiDelay: Double,
        wifiJitter: Double,
        cellDelay: Double,
        cellJitter: Double
    ): Double {
        // ABSOLUTE COMPONENT (40%)
        val absoluteReward = (
            -0.12 * (p95Delay / 100.0) +
            -0.12 * (p95Jitter / 50.0) +
            0.08 * (throughput / 10.0) +
            0.08 * max(0.0, 1.0 - lossRate * 20)
        )

        // RELATIVE COMPONENT (60%)
        val minDelay = min(wifiDelay, cellDelay)
        val minJitter = min(wifiJitter, cellJitter)
        val relativeDelay = max(0.0, p95Delay - minDelay)
        val relativeJitter = max(0.0, p95Jitter - minJitter)

        val relativeReward = (
            -0.20 * (relativeDelay / 75.0) +
            -0.20 * (relativeJitter / 40.0)
        )

        // FIXED PENALTIES (20%)
        val fixedPenalties = -0.10 * (lossRate * 10.0)

        return absoluteReward + relativeReward + fixedPenalties
    }

    private fun closeBaselineCsvLogging() {
        try {
            baselineCsvWriter?.close()
            baselineCsvWriter = null
            Log.d("Baseline", "CSV logging closed")
        } catch (e: Exception) {
            Log.e("Baseline", "Failed to close CSV: ${e.message}")
        }
    }

    // ============================================================
    // CSV Logging (for RL Training)
    // ============================================================

    private fun initializeCsvLogging() {
        try {
            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val fileName = "kestrel_training_$timestamp.csv"
            val dir = getExternalFilesDir(null)
            val file = File(dir, fileName)
            csvFilePath = file.absolutePath

            csvWriter = BufferedWriter(FileWriter(file))
            csvWriter?.write(
                "timestamp,episode,step,intent," +
                "wifi_srtt,wifi_jitter,wifi_burst,wifi_loss,wifi_throughput,wifi_queue,wifi_available," +
                "cell_srtt,cell_jitter,cell_burst,cell_loss,cell_throughput,cell_queue,cell_available," +
                "action_w_delay,action_w_jitter,action_w_loss,action_w_throughput,action_use_wifi,action_use_duplication," +
                "wifi_packets,cell_packets,p95_delay,p95_jitter,loss_rate,throughput,reward,cumulative_reward\n"
            )
            csvWriter?.flush()
            Log.d("CSV", "CSV logging initialized: $csvFilePath")
        } catch (e: Exception) {
            Log.e("CSV", "Failed to initialize CSV: ${e.message}")
        }
    }

    private fun logStepToCsv(reward: Double) {
        try {
            val action = currentAction ?: return
            val p95Delay = calculateP95(stepRttSamples)
            val p95Jitter = calculateP95(stepJitterSamples)
            val stepTime = (System.currentTimeMillis() - stepStartTime) / 1000.0
            val throughput = if (stepTime > 0) (stepBytesSent * 8.0) / (stepTime * 1_000_000.0) else 0.0
            val lossRate = if (stepTotalPackets > 0) stepLostPackets.toDouble() / stepTotalPackets else 0.0

            val line = "${System.currentTimeMillis()},$currentEpisode,$currentStep,$selectedIntent," +
                "${wifiRTT.format(2)},${wifiJitter.format(2)},${wifiBurst.format(3)},${wifiLoss.format(4)}," +
                "${wifiThroughput.format(2)},$wifiQueueDepth,${if (wifiNetwork != null) 1 else 0}," +
                "${cellularRTT.format(2)},${cellularJitter.format(2)},${cellularBurst.format(3)},${cellularLoss.format(4)}," +
                "${cellularThroughput.format(2)},$cellularQueueDepth,${if (cellularNetwork != null) 1 else 0}," +
                "${action.weightDelay.format(3)},${action.weightJitter.format(3)},${action.weightLoss.format(3)}," +
                "${action.weightThroughput.format(3)},${action.useWifi.format(3)},${action.useDuplication.format(3)}," +
                "$stepWifiPackets,$stepCellPackets,${p95Delay.format(2)},${p95Jitter.format(2)}," +
                "${lossRate.format(4)},${throughput.format(2)},${reward.format(4)},${episodeReward.format(4)}\n"

            csvWriter?.write(line)
            csvWriter?.flush()
        } catch (e: Exception) {
            Log.e("CSV", "Failed to log step: ${e.message}")
        }
    }

    private fun Double.format(digits: Int) = "%.${digits}f".format(this)

    // ============================================================
    // RL Training Mode
    // ============================================================

    private fun startRLTraining() {
        if (wifiNetwork == null || cellularNetwork == null) {
            updateStatus("ERROR: Both networks must be available!")
            return
        }

        isRunning = true
        startButton.text = "Stop Training"
        testTypeRadioGroup.isEnabled = false
        intentSpinner.isEnabled = false

        thread {
            // Connect to RL server
            updateStatus("Connecting to RL server...")
            if (!connectToRLServer()) {
                runOnUiThread {
                    updateStatus("ERROR: Cannot connect to RL server at $SERVER_IP:$RL_PORT")
                    resetUI()
                }
                return@thread
            }

            updateStatus("Connected to RL server. Starting training...")
            initializeCsvLogging()

            val powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager
            val wakeLock = powerManager.newWakeLock(
                PowerManager.PARTIAL_WAKE_LOCK,
                "KESTREL::RLTraining"
            )
            wakeLock.acquire(60 * 60 * 1000L) // 1 hour max

            try {
                // Initialize sockets
                wifiSocket = DatagramSocket().apply {
                    wifiNetwork?.bindSocket(this)
                    soTimeout = 2000
                }
                cellularSocket = DatagramSocket().apply {
                    cellularNetwork?.bindSocket(this)
                    soTimeout = 2000
                }

                val serverAddr = InetAddress.getByName(SERVER_IP)

                // Continuous training loop
                while (isRunning) {
                    runEpisode(serverAddr)
                }

            } catch (e: Exception) {
                updateStatus("ERROR: ${e.message}")
                Log.e("RL", "Training error", e)
            } finally {
                wifiSocket?.close()
                cellularSocket?.close()
                disconnectFromRLServer()
                csvWriter?.close()
                wakeLock.release()
                isRunning = false
                runOnUiThread { resetUI() }
            }
        }
    }

    private fun runEpisode(serverAddr: InetAddress) {
        currentEpisode++
        currentStep = 0
        episodeReward = 0.0

        // Reset telemetry
        wifiRTT = 100.0
        cellularRTT = 100.0
        wifiJitter = 10.0
        cellularJitter = 10.0
        wifiLoss = 0.0
        cellularLoss = 0.0
        wifiThroughput = 0.0
        cellularThroughput = 0.0
        wifiPacketsSent = 0
        wifiPacketsAcked = 0
        cellularPacketsSent = 0
        cellularPacketsAcked = 0
        wifiRttHistory.clear()
        cellularRttHistory.clear()

        val episodeStartTime = System.currentTimeMillis()
        var chunkId = 0
        val testData = ByteArray(1024 * 1024) { it.toByte() }
        val totalChunks = (testData.size + CHUNK_SIZE - 1) / CHUNK_SIZE

        updateRLStatus("Episode $currentEpisode starting...")

        while (isRunning && (System.currentTimeMillis() - episodeStartTime) < EPISODE_DURATION_MS) {
            currentStep++

            // Reset step metrics
            resetStepMetrics()

            // Get action from RL server
            currentAction = getActionFromRL()
            if (currentAction == null) {
                Log.e("RL", "Failed to get action, using defaults")
                currentAction = SchedulingAction(0.4, 0.3, 0.2, 0.1, 0.5, 0.0)
            }

            // Execute action for PACKETS_PER_STEP packets
            for (i in 0 until PACKETS_PER_STEP) {
                if (!isRunning || chunkId >= totalChunks) break

                val offset = chunkId * CHUNK_SIZE
                val length = minOf(CHUNK_SIZE, testData.size - offset)
                val chunkData = testData.copyOfRange(offset, offset + length)

                val useWifi = selectPathUsingRL(currentAction!!)
                val socket = if (useWifi) wifiSocket else cellularSocket
                val port = if (useWifi) WIFI_PORT else CELLULAR_PORT

                val packetData = ByteBuffer.allocate(12 + length).apply {
                    putInt(chunkId)
                    putLong(System.nanoTime())
                    put(chunkData)
                }.array()

                val sendTime = System.nanoTime()
                try {
                    socket?.send(DatagramPacket(packetData, packetData.size, serverAddr, port))

                    if (useWifi) {
                        stepWifiPackets++
                        wifiBytesSent += length
                    } else {
                        stepCellPackets++
                        cellularBytesSent += length
                    }
                    stepBytesSent += length
                    stepTotalPackets++

                    // Wait for ACK
                    val ackBuffer = ByteArray(12)
                    val ackPacket = DatagramPacket(ackBuffer, ackBuffer.size)
                    socket?.receive(ackPacket)

                    val recvTime = System.nanoTime()
                    val rtt = (recvTime - sendTime) / 1_000_000.0

                    // Update telemetry
                    updateJitter(rtt, useWifi)
                    updateRttHistory(rtt, useWifi)
                    updateLossRate(true, useWifi)
                    updateThroughput(length, ((recvTime - sendTime) / 1_000_000).toLong(), useWifi)

                    if (useWifi) {
                        wifiRTT = 0.875 * wifiRTT + 0.125 * rtt
                        wifiBurst = calculateBurst(true)
                    } else {
                        cellularRTT = 0.875 * cellularRTT + 0.125 * rtt
                        cellularBurst = calculateBurst(false)
                    }

                    stepRttSamples.add(rtt)

                } catch (e: Exception) {
                    // Timeout or error
                    updateLossRate(false, useWifi)
                    stepLostPackets++
                    if (useWifi) wifiRTT += 50 else cellularRTT += 50
                }

                chunkId++
                Thread.sleep(5)
            }

            // Wrap around test data
            if (chunkId >= totalChunks) chunkId = 0

            // Calculate step metrics and report reward
            val stepTime = (System.currentTimeMillis() - stepStartTime) / 1000.0
            val stepThroughput = if (stepTime > 0) (stepBytesSent * 8.0) / (stepTime * 1_000_000.0) else 0.0
            val stepLossRate = if (stepTotalPackets > 0) stepLostPackets.toDouble() / stepTotalPackets else 0.0

            val metrics = StepMetrics(
                p95Delay = calculateP95(stepRttSamples),
                p95Jitter = calculateP95(stepJitterSamples),
                lossRate = stepLossRate,
                throughput = stepThroughput,
                stallCount = 0,
                bytesSent = stepBytesSent,
                completionTime = stepTime,
                wifiPackets = stepWifiPackets,
                cellPackets = stepCellPackets
            )

            val isEpisodeDone = (System.currentTimeMillis() - episodeStartTime) >= EPISODE_DURATION_MS
            val reward = reportRewardToRL(metrics, isEpisodeDone)
            episodeReward += reward

            // Log to CSV
            logStepToCsv(reward)

            // Update UI
            updateRLStatus("""
                Episode: $currentEpisode | Step: $currentStep
                Intent: $selectedIntent

                State:
                  WiFi: ${wifiRTT.toInt()}ms, Jitter: ${wifiJitter.toInt()}ms, Loss: ${"%.2f".format(wifiLoss * 100)}%
                  Cell: ${cellularRTT.toInt()}ms, Jitter: ${cellularJitter.toInt()}ms, Loss: ${"%.2f".format(cellularLoss * 100)}%

                Action:
                  Weights: D=${"%.0f".format(currentAction!!.weightDelay * 100)}% J=${"%.0f".format(currentAction!!.weightJitter * 100)}% L=${"%.0f".format(currentAction!!.weightLoss * 100)}% T=${"%.0f".format(currentAction!!.weightThroughput * 100)}%
                  WiFi Pref: ${"%.0f".format(currentAction!!.useWifi * 100)}%

                Reward: ${"%.3f".format(reward)} | Episode: ${"%.2f".format(episodeReward)}
                Best Episode: ${"%.2f".format(bestEpisodeReward)}

                WiFi: ${wifiBytesSent / 1024}KB | Cell: ${cellularBytesSent / 1024}KB
            """.trimIndent())
        }

        // Episode done - notify server
        val response = notifyEpisodeDone()
        if (response != null) {
            if (response.isBest) {
                bestEpisodeReward = response.totalReward
            }
            updateStatus("Episode $currentEpisode done! Reward: ${"%.2f".format(response.totalReward)} " +
                    "(Best: ${if (response.isBest) "NEW!" else "%.2f".format(bestEpisodeReward)})")
        }

        // Reset for next episode
        wifiBytesSent = 0
        cellularBytesSent = 0
    }

    private fun resetStepMetrics() {
        stepWifiPackets = 0
        stepCellPackets = 0
        stepRttSamples.clear()
        stepJitterSamples.clear()
        stepBytesSent = 0
        stepStartTime = System.currentTimeMillis()
        stepLostPackets = 0
        stepTotalPackets = 0
    }

    // ============================================================
    // Original Test Modes (Min RTT / Round Robin)
    // ============================================================

    private fun startMultipathTest() {
        if (wifiNetwork == null || cellularNetwork == null) {
            updateStatus("ERROR: Both networks must be available!")
            return
        }

        isRunning = true
        startButton.text = "Stop Test"
        testTypeRadioGroup.isEnabled = false

        val powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager
        val wakeLock = powerManager.newWakeLock(
            PowerManager.PARTIAL_WAKE_LOCK,
            "Multipath::Test"
        )
        wakeLock.acquire(10 * 60 * 1000L)

        thread {
            try {
                wifiSocket = DatagramSocket().apply {
                    wifiNetwork?.bindSocket(this)
                    soTimeout = 2000
                }

                cellularSocket = DatagramSocket().apply {
                    cellularNetwork?.bindSocket(this)
                    soTimeout = 2000
                }

                val serverAddr = InetAddress.getByName(SERVER_IP)

                updateStatus("Starting multipath transfer with ${selectedTestType.name}...")

                val testData = ByteArray(1024 * 1024) { it.toByte() }
                val totalChunks = (testData.size + CHUNK_SIZE - 1) / CHUNK_SIZE

                var wifiCount = 0
                var cellularCount = 0
                var wifiBytes = 0L
                var cellularBytes = 0L
                val startTime = System.currentTimeMillis()

                for (chunkId in 0 until totalChunks) {
                    if (!isRunning) break

                    val offset = chunkId * CHUNK_SIZE
                    val length = minOf(CHUNK_SIZE, testData.size - offset)
                    val chunkData = testData.copyOfRange(offset, offset + length)

                    val useWifi = when (selectedTestType) {
                        TestType.MIN_RTT -> wifiRTT <= cellularRTT
                        TestType.ROUND_ROBIN -> chunkId % 2 == 0
                        TestType.WEIGHTED_RR -> selectPathWeightedRR()
                        TestType.RL_TRAINING -> true // Should not reach here
                    }

                    val socket = if (useWifi) wifiSocket else cellularSocket
                    val port = if (useWifi) WIFI_PORT else CELLULAR_PORT
                    val pathName = if (useWifi) "WiFi" else "Cellular"

                    val packetData = ByteBuffer.allocate(12 + length).apply {
                        putInt(chunkId)
                        putLong(System.nanoTime())
                        put(chunkData)
                    }.array()

                    val sendTime = System.nanoTime()
                    socket?.send(DatagramPacket(packetData, packetData.size, serverAddr, port))

                    if (useWifi) {
                        wifiCount++
                        wifiBytes += length
                    } else {
                        cellularCount++
                        cellularBytes += length
                    }

                    val ackBuffer = ByteArray(12)
                    val ackPacket = DatagramPacket(ackBuffer, ackBuffer.size)

                    try {
                        socket?.receive(ackPacket)
                        val recvTime = System.nanoTime()
                        val rtt = (recvTime - sendTime) / 1_000_000.0

                        if (useWifi) {
                            wifiRTT = 0.875 * wifiRTT + 0.125 * rtt
                        } else {
                            cellularRTT = 0.875 * cellularRTT + 0.125 * rtt
                        }

                        if (chunkId % 10 == 0) {
                            val progress = (chunkId * 100) / totalChunks
                            updateStatus("""
                                Progress: $progress% ($chunkId/$totalChunks)
                                Path: $pathName | RTT: ${rtt.toInt()}ms
                                WiFi RTT: ${wifiRTT.toInt()}ms | Cell RTT: ${cellularRTT.toInt()}ms
                                WiFi: $wifiCount pkts | Cell: $cellularCount pkts
                            """.trimIndent())
                        }

                    } catch (e: Exception) {
                        Log.e("Multipath", "Timeout/Error on chunk $chunkId: ${e.message}")
                        if (useWifi) wifiRTT += 50 else cellularRTT += 50
                    }

                    Thread.sleep(5)
                }

                val totalTime = (System.currentTimeMillis() - startTime) / 1000.0
                val totalBytes = wifiBytes + cellularBytes
                val throughput = (totalBytes * 8) / (totalTime * 1_000_000)

                updateStatus("""
                    COMPLETED
                    Time: ${totalTime.toInt()}s | Throughput: ${"%.2f".format(throughput)} Mbps
                    WiFi: $wifiCount pkts (${wifiBytes / 1024}KB)
                    Cellular: $cellularCount pkts (${cellularBytes / 1024}KB)
                    Final WiFi RTT: ${wifiRTT.toInt()}ms
                    Final Cell RTT: ${cellularRTT.toInt()}ms
                """.trimIndent())

            } catch (e: Exception) {
                updateStatus("ERROR: ${e.message}")
                Log.e("Multipath", "Error in transfer", e)
            } finally {
                wifiSocket?.close()
                cellularSocket?.close()
                wakeLock.release()
                isRunning = false
                runOnUiThread { resetUI() }
            }
        }
    }

    private fun stopMultipathTest() {
        isRunning = false
        updateStatus("Stopping...")
    }

    // ============================================================
    // UI Helpers
    // ============================================================

    private fun resetUI() {
        startButton.text = "Start Test"
        testTypeRadioGroup.isEnabled = true
        intentSpinner.isEnabled = true
        checkNetworksAndEnableButton()
    }

    private fun updateNetworkStatus() {
        val wifiStatus = if (wifiNetwork != null) "available" else "unavailable"
        val cellularStatus = if (cellularNetwork != null) "available" else "unavailable"
        updateStatus("WiFi: $wifiStatus, Cellular: $cellularStatus")
    }

    private fun updateStatus(text: String) {
        runOnUiThread {
            statusText.text = text
            Log.d("Multipath", text)
        }
    }

    private fun updateRLStatus(text: String) {
        runOnUiThread {
            rlStatusText.text = text
            Log.d("RL", text)
        }
    }
}
