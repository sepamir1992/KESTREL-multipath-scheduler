package com.example.multipathclient

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.net.*
import android.os.Bundle
import android.os.PowerManager
import android.util.Log
import android.widget.Button
import android.widget.RadioGroup
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetAddress
import java.nio.ByteBuffer
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity() {
    private lateinit var statusText: TextView
    private lateinit var startButton: Button
    private lateinit var testTypeRadioGroup: RadioGroup
    private lateinit var connectivityManager: ConnectivityManager

    private var wifiNetwork: Network? = null
    private var cellularNetwork: Network? = null
    private var wifiSocket: DatagramSocket? = null
    private var cellularSocket: DatagramSocket? = null

    private var wifiRTT = 100.0 // ms
    private var cellularRTT = 100.0 // ms
    private var isRunning = false

    private enum class TestType { MIN_RTT, ROUND_ROBIN }
    private var selectedTestType = TestType.MIN_RTT

    private val SERVER_IP = "34.45.243.172" // Public IP of your cloud server
    private val WIFI_PORT = 5000
    private val CELLULAR_PORT = 5001
    private val CHUNK_SIZE = 1400

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        statusText = findViewById(R.id.statusText)
        startButton = findViewById(R.id.startButton)
        testTypeRadioGroup = findViewById(R.id.testTypeRadioGroup)

        startButton.isEnabled = false

        connectivityManager = getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_NETWORK_STATE)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                arrayOf(Manifest.permission.ACCESS_NETWORK_STATE,
                    Manifest.permission.CHANGE_NETWORK_STATE,
                    Manifest.permission.INTERNET), 100)
        }

        testTypeRadioGroup.setOnCheckedChangeListener { _, checkedId ->
            selectedTestType = when (checkedId) {
                R.id.roundRobinRadioButton -> TestType.ROUND_ROBIN
                else -> TestType.MIN_RTT
            }
        }

        startButton.setOnClickListener {
            if (!isRunning) {
                startMultipathTest()
            } else {
                stopMultipathTest()
            }
        }

        setupNetworks()
    }

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
                    ✓ COMPLETED
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
                runOnUiThread {
                    startButton.text = "Start Test"
                    testTypeRadioGroup.isEnabled = true
                    checkNetworksAndEnableButton()
                }
            }
        }
    }

    private fun stopMultipathTest() {
        isRunning = false
        updateStatus("Stopping...")
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