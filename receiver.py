import socket
import time

# TCP Server - Run this on the RECEIVING computer
# This listens for incoming connections and shows throughput

HOST = '0.0.0.0'  # Listen on all network interfaces
PORT = 8080       # Port to listen on

# Create TCP socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print(f"Server listening on port {PORT}...")
print("Waiting for connection...")

# Wait for a connection
conn, addr = server_socket.accept()
print(f"Connected by {addr[0]}:{addr[1]}")

# Receive data
start_time = time.time()
data = conn.recv(1024)
end_time = time.time()

if data:
    # Calculate metrics
    data_size = len(data)
    transfer_time = end_time - start_time
    throughput_bps = (data_size * 8) / transfer_time if transfer_time > 0 else 0
    throughput_kbps = throughput_bps / 1000
    throughput_mbps = throughput_bps / 1_000_000
    
    print("\n" + "="*50)
    print("RECEIVED DATA")
    print("="*50)
    print(f"Message: {data.decode()}")
    print(f"Data size: {data_size} bytes")
    print(f"Transfer time: {transfer_time:.6f} seconds")
    print(f"Throughput: {throughput_bps:.2f} bits/sec")
    print(f"Throughput: {throughput_kbps:.2f} Kbps")
    print(f"Throughput: {throughput_mbps:.6f} Mbps")
    print(f"Remote address: {addr[0]}:{addr[1]}")
    print("="*50 + "\n")

# Clean up
conn.close()
server_socket.close()
print("Connection closed.")
