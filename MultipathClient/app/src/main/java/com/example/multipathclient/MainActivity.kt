package com.example.multipathclient

import android.Manifest
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.net.*
import android.os.Build
import android.os.Bundle
import android.os.IBinder
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

class MainActivity : AppCompatActivity(), RLTrainingService.TrainingCallback {
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

    // RTT tracking for non-RL modes
    private var wifiRTT = 100.0
    private var cellularRTT = 100.0

    private var isRunning = false
    private var selectedIntent = "video_streaming"

    private enum class TestType { MIN_RTT, ROUND_ROBIN, RL_TRAINING }
    private var selectedTestType = TestType.MIN_RTT

    private val SERVER_IP = "35.225.121.173"
    private val WIFI_PORT = 5000
    private val CELLULAR_PORT = 5001
    private val CHUNK_SIZE = 1400

    // Service binding
    private var rlService: RLTrainingService? = null
    private var serviceBound = false

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            val binder = service as RLTrainingService.LocalBinder
            rlService = binder.getService()
            rlService?.callback = this@MainActivity
            serviceBound = true
            Log.d("MainActivity", "Service connected")
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            rlService?.callback = null
            rlService = null
            serviceBound = false
            Log.d("MainActivity", "Service disconnected")
        }
    }

    // ============================================================
    // RLTrainingService.TrainingCallback implementation
    // ============================================================

    override fun onStatusUpdate(status: String) {
        runOnUiThread {
            statusText.text = status
        }
    }

    override fun onRLStatusUpdate(rlStatus: String) {
        runOnUiThread {
            rlStatusText.text = rlStatus
        }
    }

    override fun onTrainingStopped() {
        runOnUiThread {
            isRunning = false
            resetUI()
        }
    }

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
        requestPermissions()

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
                if (selectedTestType == TestType.RL_TRAINING) {
                    stopRLTraining()
                } else {
                    stopMultipathTest()
                }
            }
        }

        setupNetworks()
    }

    override fun onStart() {
        super.onStart()
        // Bind to RLTrainingService if it's running
        Intent(this, RLTrainingService::class.java).also { intent ->
            bindService(intent, serviceConnection, Context.BIND_AUTO_CREATE)
        }
    }

    override fun onStop() {
        super.onStop()
        // Unbind from service (service continues running)
        if (serviceBound) {
            rlService?.callback = null
            unbindService(serviceConnection)
            serviceBound = false
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        // Only stop non-RL tests
        if (selectedTestType != TestType.RL_TRAINING) {
            stopMultipathTest()
        }
    }

    private fun requestPermissions() {
        val permissions = mutableListOf(
            Manifest.permission.ACCESS_NETWORK_STATE,
            Manifest.permission.CHANGE_NETWORK_STATE,
            Manifest.permission.INTERNET,
            Manifest.permission.WAKE_LOCK
        )

        // Android 13+ requires POST_NOTIFICATIONS permission
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            permissions.add(Manifest.permission.POST_NOTIFICATIONS)
        }

        val permissionsToRequest = permissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }

        if (permissionsToRequest.isNotEmpty()) {
            ActivityCompat.requestPermissions(this, permissionsToRequest.toTypedArray(), 100)
        }
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
    // RL Training Mode (via Foreground Service)
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

        // Start the foreground service
        val serviceIntent = Intent(this, RLTrainingService::class.java).apply {
            action = RLTrainingService.ACTION_START
            putExtra(RLTrainingService.EXTRA_INTENT, selectedIntent)
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(serviceIntent)
        } else {
            startService(serviceIntent)
        }

        // Bind to get updates
        bindService(serviceIntent, serviceConnection, Context.BIND_AUTO_CREATE)

        updateStatus("Starting RL Training Service...")
    }

    private fun stopRLTraining() {
        rlService?.stopTraining()
        isRunning = false
        resetUI()
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
}
