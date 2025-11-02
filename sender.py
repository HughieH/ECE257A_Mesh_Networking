import socket
import time

# TCP Client - Run this on the SENDING computer
# This connects to the receiver and sends "hello world"

HOST = '192.168.168.10'  # IP address of the REMOTE pMDDL radio
PORT = 8080              # Same port as receiver

# Message to send
message = "hello world"
message_bytes = message.encode()

# Create TCP socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

print(f"Connecting to {HOST}:{PORT}...")
connection_start = time.time()
client_socket.connect((HOST, PORT))
connection_end = time.time()
connection_time = connection_end - connection_start

print(f"Connected! (Connection time: {connection_time:.6f} seconds)")

# Send the message and measure time
send_start = time.time()
bytes_sent = client_socket.send(message_bytes)
send_end = time.time()
send_time = send_end - send_start

# Calculate metrics
total_time = send_end - connection_start
throughput_bps = (bytes_sent * 8) / send_time if send_time > 0 else 0
throughput_kbps = throughput_bps / 1000
throughput_mbps = throughput_bps / 1_000_000

print("\n" + "="*50)
print("TRANSMISSION COMPLETE")
print("="*50)
print(f"Message sent: {message}")
print(f"Data size: {bytes_sent} bytes")
print(f"Connection time: {connection_time:.6f} seconds")
print(f"Send time: {send_time:.6f} seconds")
print(f"Total time: {total_time:.6f} seconds")
print(f"Throughput: {throughput_bps:.2f} bits/sec")
print(f"Throughput: {throughput_kbps:.2f} Kbps")
print(f"Throughput: {throughput_mbps:.6f} Mbps")
print("="*50 + "\n")

# Clean up
client_socket.close()
print("Connection closed.")

