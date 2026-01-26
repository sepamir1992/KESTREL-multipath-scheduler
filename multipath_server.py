#!/usr/bin/env python3
"""
Multipath Server for Windows
Receives packets from WiFi and Cellular paths simultaneously
"""

import socket
import struct
import time
import threading
from collections import defaultdict
from datetime import datetime

class MultipathServer:
    def __init__(self, wifi_port=5000, cellular_port=5001):
        self.wifi_port = wifi_port
        self.cellular_port = cellular_port
        
        # Statistics
        self.wifi_packets = 0
        self.cellular_packets = 0
        self.wifi_bytes = 0
        self.cellular_bytes = 0
        self.received_chunks = defaultdict(lambda: {'time': 0, 'path': ''})
        
        self.start_time = None
        self.running = True
        
    def handle_path(self, sock, path_name):
        """Handle incoming packets on a specific path"""
        print(f"[{path_name}] Listening on port {sock.getsockname()[1]}...")
        
        while self.running:
            try:
                data, addr = sock.recvfrom(65536)
                recv_time = time.time()
                
                if self.start_time is None:
                    self.start_time = recv_time
                
                # Parse packet: [chunkId(4)][timestamp(8)][data...]
                if len(data) < 12:
                    continue
                    
                chunk_id = struct.unpack('!I', data[0:4])[0]
                send_timestamp = struct.unpack('!Q', data[4:12])[0]
                chunk_data = data[12:]
                
                # Record statistics
                if path_name == "WiFi":
                    self.wifi_packets += 1
                    self.wifi_bytes += len(chunk_data)
                else:
                    self.cellular_packets += 1
                    self.cellular_bytes += len(chunk_data)
                
                # Track this chunk
                if chunk_id not in self.received_chunks:
                    self.received_chunks[chunk_id] = {
                        'time': recv_time,
                        'path': path_name,
                        'size': len(chunk_data)
                    }
                
                # Note: RTT is measured on client side (ACK round-trip)
                # Server just logs receive time for reference
                rtt_ms = 0.0  # Actual RTT measured by client
                
                # Send ACK back: [chunkId(4)][recvTime(8)]
                ack_data = struct.pack('!IQ', chunk_id, int(time.time_ns()))
                sock.sendto(ack_data, addr)
                
                # Log every 10 packets
                if chunk_id % 10 == 0:
                    elapsed = recv_time - self.start_time if self.start_time else 0
                    total_packets = self.wifi_packets + self.cellular_packets
                    total_bytes = self.wifi_bytes + self.cellular_bytes
                    
                    throughput_mbps = (total_bytes * 8) / (elapsed * 1_000_000) if elapsed > 0 else 0
                    
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Chunk {chunk_id} from {addr[0]} via {path_name}")
                    print(f"  Size: {len(chunk_data)} bytes")
                    print(f"  Total: {total_packets} pkts | {total_bytes/1024:.0f} KB | {throughput_mbps:.2f} Mbps")
                    print(f"  WiFi: {self.wifi_packets} pkts ({self.wifi_bytes/1024:.0f} KB)")
                    print(f"  Cell: {self.cellular_packets} pkts ({self.cellular_bytes/1024:.0f} KB)")
                
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[{path_name}] Error: {e}")
                
    def start(self):
        """Start both path listeners"""
        # Create WiFi socket
        wifi_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        wifi_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        wifi_sock.bind(('0.0.0.0', self.wifi_port))
        wifi_sock.settimeout(1.0)
        
        # Create Cellular socket
        cellular_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        cellular_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        cellular_sock.bind(('0.0.0.0', self.cellular_port))
        cellular_sock.settimeout(1.0)
        
        # Get local IP
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        
        print("=" * 60)
        print("MULTIPATH SERVER STARTED")
        print("=" * 60)
        print(f"Server IP: {local_ip}")
        print(f"WiFi Port: {self.wifi_port}")
        print(f"Cellular Port: {self.cellular_port}")
        print("\nIMPORTANT: Update SERVER_IP in Android app to: {local_ip}")
        print("\nMake sure:")
        print("  1. Windows Firewall allows UDP ports 5000-5001")
        print("  2. Phone and laptop are on same network (for WiFi path)")
        print("  3. Phone has cellular data enabled")
        print("=" * 60)
        print("\nWaiting for packets...\n")
        
        # Start threads for each path
        wifi_thread = threading.Thread(
            target=self.handle_path, 
            args=(wifi_sock, "WiFi"),
            daemon=True
        )
        
        cellular_thread = threading.Thread(
            target=self.handle_path,
            args=(cellular_sock, "Cellular"),
            daemon=True
        )
        
        wifi_thread.start()
        cellular_thread.start()
        
        try:
            # Keep main thread alive
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n" + "=" * 60)
            print("STOPPING SERVER...")
            print("=" * 60)
            self.running = False
            
            # Print final statistics
            total_time = time.time() - self.start_time if self.start_time else 0
            total_packets = self.wifi_packets + self.cellular_packets
            total_bytes = self.wifi_bytes + self.cellular_bytes
            
            if total_time > 0:
                throughput_mbps = (total_bytes * 8) / (total_time * 1_000_000)
                
                print(f"\nFINAL STATISTICS:")
                print(f"  Duration: {total_time:.2f} seconds")
                print(f"  Total Packets: {total_packets}")
                print(f"  Total Data: {total_bytes/1024:.2f} KB ({total_bytes/1024/1024:.2f} MB)")
                print(f"  Average Throughput: {throughput_mbps:.2f} Mbps")
                print(f"\n  WiFi Path:")
                print(f"    Packets: {self.wifi_packets} ({self.wifi_packets*100/total_packets:.1f}%)")
                print(f"    Data: {self.wifi_bytes/1024:.2f} KB")
                print(f"\n  Cellular Path:")
                print(f"    Packets: {self.cellular_packets} ({self.cellular_packets*100/total_packets:.1f}%)")
                print(f"    Data: {self.cellular_bytes/1024:.2f} KB")
                print(f"\n  Unique Chunks Received: {len(self.received_chunks)}")
            
            print("=" * 60)
        
        finally:
            wifi_sock.close()
            cellular_sock.close()

def main():
    print("\n" + "=" * 60)
    print("  MULTIPATH RESEARCH SERVER")
    print("  MinRTT Scheduler Evaluation")
    print("=" * 60 + "\n")
    
    server = MultipathServer(wifi_port=5000, cellular_port=5001)
    server.start()

if __name__ == "__main__":
    main()