package com.example.multipathclient

import android.app.*
import android.content.Context
import android.content.Intent
import android.net.*
import android.os.Binder
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import androidx.core.app.NotificationCompat
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

class RLTrainingService : Service() {

    companion object {
        const val CHANNEL_ID = "RLTrainingChannel"
        const val NOTIFICATION_ID = 1
        const val ACTION_START = "com.example.multipathclient.START"
        const val ACTION_STOP = "com.example.multipathclient.STOP"
        const val EXTRA_INTENT = "intent"

        private const val TAG = "RLTrainingService"
    }

    // Binder for activity communication
    private val binder = LocalBinder()

    inner class LocalBinder : Binder() {
        fun getService(): RLTrainingService = this@RLTrainingService
    }

    // Callback interface for UI updates
    interface TrainingCallback {
        fun onStatusUpdate(status: String)
        fun onRLStatusUpdate(rlStatus: String)
        fun onTrainingStopped()
    }

    var callback: TrainingCallback? = null

    // Network and socket variables
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

    private val SERVER_IP = "35.225.121.173"
    private val WIFI_PORT = 5000
    private val CELLULAR_PORT = 5001
    private val RL_PORT = 6000
    private val CHUNK_SIZE = 1400
    private val PACKETS_PER_STEP = 10
    private val EPISODE_DURATION_MS = 60_000L

    // CSV Logging
    private var csvWriter: BufferedWriter? = null

    // Wake lock
    private var wakeLock: PowerManager.WakeLock? = null

    // Data classes (same as MainActivity)
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

    override fun onCreate() {
        super.onCreate()
        connectivityManager = getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        createNotificationChannel()
        acquireWakeLock()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START -> {
                selectedIntent = intent.getStringExtra(EXTRA_INTENT) ?: "video_streaming"
                startForeground(NOTIFICATION_ID, createNotification("Starting RL Training..."))
                startTraining()
            }
            ACTION_STOP -> {
                stopTraining()
            }
        }
        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder {
        return binder
    }

    override fun onDestroy() {
        stopTraining()
        releaseWakeLock()
        super.onDestroy()
    }

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID,
            "RL Training Service",
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            description = "KESTREL RL Training running in background"
            setShowBadge(false)
        }
        val notificationManager = getSystemService(NotificationManager::class.java)
        notificationManager.createNotificationChannel(channel)
    }

    private fun createNotification(text: String): Notification {
        val stopIntent = Intent(this, RLTrainingService::class.java).apply {
            action = ACTION_STOP
        }
        val stopPendingIntent = PendingIntent.getService(
            this, 0, stopIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val openIntent = Intent(this, MainActivity::class.java)
        val openPendingIntent = PendingIntent.getActivity(
            this, 0, openIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("KESTREL RL Training")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_menu_rotate)
            .setOngoing(true)
            .setContentIntent(openPendingIntent)
            .addAction(android.R.drawable.ic_media_pause, "Stop", stopPendingIntent)
            .build()
    }

    private fun updateNotification(text: String) {
        val notificationManager = getSystemService(NotificationManager::class.java)
        notificationManager.notify(NOTIFICATION_ID, createNotification(text))
    }

    private fun acquireWakeLock() {
        val powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = powerManager.newWakeLock(
            PowerManager.PARTIAL_WAKE_LOCK,
            "KESTREL::RLTrainingService"
        ).apply {
            acquire(4 * 60 * 60 * 1000L) // 4 hours max
        }
        Log.d(TAG, "Wake lock acquired")
    }

    private fun releaseWakeLock() {
        wakeLock?.let {
            if (it.isHeld) {
                it.release()
                Log.d(TAG, "Wake lock released")
            }
        }
        wakeLock = null
    }

    // ============================================================
    // Network Setup
    // ============================================================

    private fun setupNetworks(): Boolean {
        var wifiReady = false
        var cellularReady = false
        val lock = Object()

        val wifiRequest = NetworkRequest.Builder()
            .addTransportType(NetworkCapabilities.TRANSPORT_WIFI)
            .build()

        connectivityManager.requestNetwork(wifiRequest, object : ConnectivityManager.NetworkCallback() {
            override fun onAvailable(network: Network) {
                wifiNetwork = network
                synchronized(lock) {
                    wifiReady = true
                    lock.notifyAll()
                }
            }
        })

        val cellularRequest = NetworkRequest.Builder()
            .addTransportType(NetworkCapabilities.TRANSPORT_CELLULAR)
            .build()

        connectivityManager.requestNetwork(cellularRequest, object : ConnectivityManager.NetworkCallback() {
            override fun onAvailable(network: Network) {
                cellularNetwork = network
                synchronized(lock) {
                    cellularReady = true
                    lock.notifyAll()
                }
            }
        })

        // Wait up to 10 seconds for networks
        synchronized(lock) {
            val timeout = System.currentTimeMillis() + 10_000
            while ((!wifiReady || !cellularReady) && System.currentTimeMillis() < timeout) {
                lock.wait(1000)
            }
        }

        return wifiReady && cellularReady
    }

    // ============================================================
    // Training Logic
    // ============================================================

    private fun startTraining() {
        if (isRunning) return
        isRunning = true

        thread {
            updateStatus("Setting up networks...")

            if (!setupNetworks()) {
                updateStatus("ERROR: Could not get both networks")
                stopSelf()
                return@thread
            }

            updateStatus("Connecting to RL server...")
            if (!connectToRLServer()) {
                updateStatus("ERROR: Cannot connect to RL server at $SERVER_IP:$RL_PORT")
                stopSelf()
                return@thread
            }

            updateStatus("Connected! Starting training...")
            initializeCsvLogging()

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

                while (isRunning) {
                    runEpisode(serverAddr)
                }

            } catch (e: Exception) {
                updateStatus("ERROR: ${e.message}")
                Log.e(TAG, "Training error", e)
            } finally {
                cleanup()
            }
        }
    }

    fun stopTraining() {
        isRunning = false
        cleanup()
        callback?.onTrainingStopped()
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    private fun cleanup() {
        wifiSocket?.close()
        cellularSocket?.close()
        disconnectFromRLServer()
        csvWriter?.close()
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

        updateNotification("Episode $currentEpisode | $selectedIntent")
        updateRLStatus("Episode $currentEpisode starting...")

        while (isRunning && (System.currentTimeMillis() - episodeStartTime) < EPISODE_DURATION_MS) {
            currentStep++
            resetStepMetrics()

            currentAction = getActionFromRL()
            if (currentAction == null) {
                Log.e(TAG, "Failed to get action, using defaults")
                currentAction = SchedulingAction(0.4, 0.3, 0.2, 0.1, 0.5, 0.0)
            }

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

                    val ackBuffer = ByteArray(12)
                    val ackPacket = DatagramPacket(ackBuffer, ackBuffer.size)
                    socket?.receive(ackPacket)

                    val recvTime = System.nanoTime()
                    val rtt = (recvTime - sendTime) / 1_000_000.0

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
                    updateLossRate(false, useWifi)
                    stepLostPackets++
                    if (useWifi) wifiRTT += 50 else cellularRTT += 50
                }

                chunkId++
                Thread.sleep(5)
            }

            if (chunkId >= totalChunks) chunkId = 0

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

            logStepToCsv(reward)

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

        val response = notifyEpisodeDone()
        if (response != null) {
            if (response.isBest) {
                bestEpisodeReward = response.totalReward
            }
            updateStatus("Episode $currentEpisode done! Reward: ${"%.2f".format(response.totalReward)} " +
                    "(Best: ${if (response.isBest) "NEW!" else "%.2f".format(bestEpisodeReward)})")
            updateNotification("Ep $currentEpisode done | Reward: ${"%.2f".format(response.totalReward)}")
        }

        wifiBytesSent = 0
        cellularBytesSent = 0
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
        val mbps = (bytes * 8.0) / (timeMs * 1000.0)
        if (isWifi) {
            wifiThroughput = 0.875 * wifiThroughput + 0.125 * mbps
        } else {
            cellularThroughput = 0.875 * cellularThroughput + 0.125 * mbps
        }
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
    // RL Communication
    // ============================================================

    private fun connectToRLServer(): Boolean {
        return try {
            rlSocket = Socket(SERVER_IP, RL_PORT)
            rlSocket?.soTimeout = 5000
            rlReader = BufferedReader(InputStreamReader(rlSocket!!.getInputStream()))
            rlWriter = PrintWriter(BufferedWriter(OutputStreamWriter(rlSocket!!.getOutputStream())), true)
            Log.d(TAG, "Connected to RL server")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to connect to RL server: ${e.message}")
            false
        }
    }

    private fun disconnectFromRLServer() {
        try {
            rlReader?.close()
            rlWriter?.close()
            rlSocket?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error closing RL connection: ${e.message}")
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
            Log.e(TAG, "RL communication error: ${e.message}")
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
                Log.e(TAG, "Failed to parse action response: ${e.message}")
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
                Log.e(TAG, "Failed to parse reward response: ${e.message}")
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
                Log.e(TAG, "Failed to parse episode done response: ${e.message}")
                null
            }
        } else null
    }

    // ============================================================
    // Path Scoring
    // ============================================================

    private fun calculatePathScore(
        srtt: Double,
        jitter: Double,
        loss: Double,
        throughput: Double,
        action: SchedulingAction
    ): Double {
        val normDelay = srtt / 200.0
        val normJitter = jitter / 50.0
        val normLoss = loss
        val normThroughput = 1.0 - (throughput / 100.0)

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
            Math.random() < action.useWifi
        } else {
            wifiScore < cellScore
        }
    }

    // ============================================================
    // CSV Logging
    // ============================================================

    private fun initializeCsvLogging() {
        try {
            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val fileName = "kestrel_training_$timestamp.csv"
            val dir = getExternalFilesDir(null)
            val file = File(dir, fileName)

            csvWriter = BufferedWriter(FileWriter(file))
            csvWriter?.write(
                "timestamp,episode,step,intent," +
                "wifi_srtt,wifi_jitter,wifi_burst,wifi_loss,wifi_throughput,wifi_queue,wifi_available," +
                "cell_srtt,cell_jitter,cell_burst,cell_loss,cell_throughput,cell_queue,cell_available," +
                "action_w_delay,action_w_jitter,action_w_loss,action_w_throughput,action_use_wifi,action_use_duplication," +
                "wifi_packets,cell_packets,p95_delay,p95_jitter,loss_rate,throughput,reward,cumulative_reward\n"
            )
            csvWriter?.flush()
            Log.d(TAG, "CSV logging initialized: ${file.absolutePath}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize CSV: ${e.message}")
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
            Log.e(TAG, "Failed to log step: ${e.message}")
        }
    }

    private fun Double.format(digits: Int) = "%.${digits}f".format(this)

    // ============================================================
    // UI Updates
    // ============================================================

    private fun updateStatus(text: String) {
        Log.d(TAG, text)
        callback?.onStatusUpdate(text)
    }

    private fun updateRLStatus(text: String) {
        Log.d(TAG, text)
        callback?.onRLStatusUpdate(text)
    }
}
