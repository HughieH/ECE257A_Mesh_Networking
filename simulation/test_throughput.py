import numpy as np
import matplotlib.pyplot as plt

# ==============================
# Configuration parameters
# ==============================
TX_POWER = 20        # dBm - transmit power
FREQUENCY = 2.4e9    # Hz  - 2.4 GHz
NOISE_FLOOR = -90    # dBm - total noise power in band (simple model)
BANDWIDTH = 20e6     # Hz  - 20 MHz (typical Wi-Fi channel)

# ==============================
# Utility conversions
# ==============================
def dbm_to_watts(dbm: float) -> float:
    return 10 ** ((dbm - 30) / 10)

def watts_to_dbm(watts: float) -> float:
    return 10 * np.log10(watts) + 30

# ==============================
# Channel models
# ==============================
def calculate_path_loss(distance: float, frequency: float) -> float:
    """Free-space path loss (Friis) in dB."""
    if distance <= 0:
        return 0.0
    c = 3e8
    return 20 * np.log10(distance) + 20 * np.log10(frequency) + 20 * np.log10(4 * np.pi / c)

def calculate_shannon_capacity(distance: float, tx_power_dbm: float,
                               noise_floor_dbm: float, bandwidth_hz: float, frequency_hz: float) -> float:
    """Shannon capacity (bps): C = B * log2(1 + SNR)."""
    if distance <= 0:
        return float('inf')
    path_loss_db = calculate_path_loss(distance, frequency_hz)
    rx_power_dbm = tx_power_dbm - path_loss_db
    rx_power_w = dbm_to_watts(rx_power_dbm)
    noise_power_w = dbm_to_watts(noise_floor_dbm)
    snr_linear = rx_power_w / noise_power_w
    return bandwidth_hz * np.log2(1 + snr_linear)

# ==============================
# Graph / path utils
# ==============================
def find_best_path(r_matrix, capacity_matrix, start_idx, end_idx):
    """Find widest path (max-min capacity)."""
    n = len(r_matrix)
    max_capacity = np.full(n, -np.inf)
    max_capacity[start_idx] = float('inf')
    previous = [-1] * n
    visited = [False] * n

    for _ in range(n):
        current = np.argmax(np.where(~np.array(visited), max_capacity, -np.inf))
        if max_capacity[current] == -np.inf:
            break
        if current == end_idx:
            break
        visited[current] = True

        for neighbor in range(n):
            if not visited[neighbor] and r_matrix[current, neighbor] > 0:
                link_cap = capacity_matrix[current, neighbor]
                new_cap = min(max_capacity[current], link_cap)
                if new_cap > max_capacity[neighbor]:
                    max_capacity[neighbor] = new_cap
                    previous[neighbor] = current

    if max_capacity[end_idx] <= 0 or max_capacity[end_idx] == -np.inf:
        return 0.0, []
    path = []
    node = end_idx
    while node != -1:
        path.insert(0, node)
        node = previous[node]
    if not path or path[0] != start_idx:
        return 0.0, []
    return max_capacity[end_idx], path

def build_mesh_capacity_matrix(r_matrix, direct_capacity_matrix):
    """Compute widest-path capacity between all node pairs."""
    n = r_matrix.shape[0]
    mesh_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                mesh_matrix[i, j], _ = find_best_path(r_matrix, direct_capacity_matrix, i, j)
    return mesh_matrix

def get_r_matrix(real_rx_pos):
    """NxN symmetric distance matrix."""
    n = len(real_rx_pos)
    r = np.zeros((n, n))
    for i in range(n):
        x1, y1 = real_rx_pos[i][1], real_rx_pos[i][2]
        for j in range(i + 1, n):
            x2, y2 = real_rx_pos[j][1], real_rx_pos[j][2]
            d = np.hypot(x2 - x1, y2 - y1)
            r[i, j] = d
            r[j, i] = d
    return r

# ==============================
# Plotting
# ==============================
def plot_receiver_grid(real_rx_pos, r_matrix, x_max, y_max):
    """Plot receivers + pairwise distances."""
    fig = plt.figure(figsize=(10, 10))
    x = [p[1] for p in real_rx_pos]
    y = [p[2] for p in real_rx_pos]
    names = [p[0] for p in real_rx_pos]

    n = len(real_rx_pos)
    for i in range(n):
        for j in range(i + 1, n):
            plt.plot([x[i], x[j]], [y[i], y[j]], 'b-', linewidth=1.5, alpha=0.5, zorder=1)
            midx, midy = (x[i] + x[j]) / 2, (y[i] + y[j]) / 2
            plt.text(midx, midy, f'{r_matrix[i,j]:.1f}m', fontsize=9, ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow',
                               edgecolor='black', alpha=0.7), zorder=2)

    plt.scatter(x, y, s=500, c='red', marker='o', edgecolors='black', linewidths=3, zorder=3)
    for name, xi, yi in real_rx_pos:
        plt.text(xi, yi, name, fontsize=12, fontweight='bold',
                 ha='center', va='center', color='white', zorder=4)

    plt.title("Receiver Grid with Distances", fontsize=16, fontweight='bold')
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.axis('equal')
    #plt.xlim(-2, x_max + 2)
    #plt.ylim(-2, y_max + 2)
   
    plt.tight_layout()
    return fig  # return fig instead of showing

def _make_table(ax, real_rx_pos, matrix, title):
    """Helper: draw table for capacity matrix."""
    names = [p[0] for p in real_rx_pos]
    n = len(names)
    cap_mbps = np.round(matrix / 1e6, 2)
    ax.axis('off')
    ax.axis('tight')
    header = ["Node"] + names
    table_data = [[names[i]] + list(cap_mbps[i]) for i in range(n)]
    table = ax.table(cellText=table_data, colLabels=header, loc='center',
                     cellLoc='center', rowLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.15)
    for i in range(n):
        table[i + 1, i + 1].set_facecolor("#d9d9d9")
    for r in range(1, n + 1):
        for c in range(1, n + 1):
            if c != r:
                table[r, c].set_facecolor("#f7f7f7")
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)

def plot_capacity_tables(real_rx_pos, direct_capacity_matrix, mesh_capacity_matrix):
    """Plot direct and mesh capacity tables together."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6 + 0.35 * len(real_rx_pos)),
                                   constrained_layout=True)
    _make_table(ax1, real_rx_pos, direct_capacity_matrix, "Direct Link Capacity (Mbps)")
    _make_table(ax2, real_rx_pos, mesh_capacity_matrix, "Mesh (Widest-Path) Capacity (Mbps)")
    return fig  # return fig instead of showing

# ==============================
# Main
# ==============================
def main():
    x_max, y_max = 50, 50
    real_rx_pos = [
        ('A', 24, 40),
        ('B', 10, 10),
        ('C', 40, 10),
        ('D', 25, 25),
        ('E', 35, 45),
    ]

    print("Receiver Grid Configuration")
    print("=" * 70)
    print(f"Tx Power: {TX_POWER} dBm, Freq: {FREQUENCY/1e9:.2f} GHz, Bandwidth: {BANDWIDTH/1e6:.1f} MHz\n")

    r_matrix = get_r_matrix(real_rx_pos)
    n = len(real_rx_pos)

    # Direct link capacities
    direct_cap = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                direct_cap[i, j] = calculate_shannon_capacity(r_matrix[i, j],
                                                             TX_POWER, NOISE_FLOOR,
                                                             BANDWIDTH, FREQUENCY)

    # Mesh (widest-path) capacities
    mesh_cap = build_mesh_capacity_matrix(r_matrix, direct_cap)

    # Console summary
    print("Distance Matrix (m):")
    print(np.round(r_matrix, 2), "\n")
    print("Direct Capacity Matrix (Mbps):")
    print(np.round(direct_cap / 1e6, 2), "\n")
    print("Mesh Capacity Matrix (Mbps):")
    print(np.round(mesh_cap / 1e6, 2), "\n")

    # Generate figures (but don't show yet)
    fig1 = plot_receiver_grid(real_rx_pos, r_matrix, x_max, y_max)
    fig2 = plot_capacity_tables(real_rx_pos, direct_cap, mesh_cap)

    # ✅ Show all figures at once
    plt.show()

if __name__ == "__main__":
    main()