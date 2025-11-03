import numpy as np
import matplotlib.pyplot as plt



# Configuration parameters
TX_POWER = 20  # dBm - transmit power
FREQUENCY = 2.4e9  # Hz - 2.4 GHz
NOISE_FLOOR = -90  # dBm - noise floor

def create_receiver_grid(x_max, y_max, num_points, goal_receiver_positions):
    """
    Place receivers on a fine grid as accurately as possible.
    
    Args:
        x_max: Maximum x coordinate
        y_max: Maximum y coordinate
        num_points: Number of receivers
        receiver_positions: List of (name, x_desired, y_desired) tuples
    
    Returns:
        List of (name, x_actual, y_actual) tuples with grid-snapped positions
    """
    # Define fine grid space
    x_window = np.linspace(0, x_max, num_points)
    y_window = np.linspace(0, y_max, num_points)
    
    placed_receivers = []
    
    # Place each receiver on grid as accurately as possible
    for name, x_desired, y_desired in goal_receiver_positions:
        # Find closest x position
        x_index = 0
        x_error = float('inf')
        for i in range(len(x_window)):
            error = (x_desired - x_window[i]) ** 2
            if error < x_error:
                x_error = error
                x_index = i
        
        # Find closest y position
        y_index = 0
        y_error = float('inf')
        for j in range(len(y_window)):
            error = (y_desired - y_window[j]) ** 2
            if error < y_error:
                y_error = error
                y_index = j
        
        placed_receivers.append((name, x_window[x_index], y_window[y_index]))
    
    return placed_receivers

def get_r_matrix(real_rx_pos):
    """
    Create a distance matrix between all receivers.
    
    Args:
        real_rx_pos: List of (name, x, y) tuples
    
    Returns:
        numpy array: NxN matrix where element [i,j] is distance between receiver i and j
    """
    n = len(real_rx_pos)
    r_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                r_matrix[i, j] = 0
            else:
                x1, y1 = real_rx_pos[i][1], real_rx_pos[i][2]
                x2, y2 = real_rx_pos[j][1], real_rx_pos[j][2]
                distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                r_matrix[i, j] = distance
    
    return r_matrix

def plot_receiver_grid(real_rx_pos, x_max, y_max):
    """Plot the receivers in the grid with connecting lines and distances."""
    plt.figure(figsize=(12, 12))
    
    # Extract coordinates
    x_coords = [pos[1] for pos in real_rx_pos]
    y_coords = [pos[2] for pos in real_rx_pos]
    names = [pos[0] for pos in real_rx_pos]
    
    # Draw lines between all receivers with distance labels
    n = len(real_rx_pos)
    for i in range(n):
        for j in range(i + 1, n):  # Only draw each line once
            x1, y1 = real_rx_pos[i][1], real_rx_pos[i][2]
            x2, y2 = real_rx_pos[j][1], real_rx_pos[j][2]
            
            # Draw the line
            plt.plot([x1, x2], [y1, y2], 'b-', linewidth=1.5, alpha=0.5, zorder=1)
            
            # Calculate distance
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Calculate midpoint for label
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            
            # Add distance label with background box for readability
            plt.text(mid_x, mid_y, f'{distance:.1f}m', 
                    fontsize=9, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', 
                             edgecolor='black', alpha=0.7),
                    zorder=2)
    
    # Plot the receivers (on top of lines)
    plt.scatter(x_coords, y_coords, s=500, c='red', marker='o', 
                edgecolors='black', linewidths=3, zorder=3, alpha=0.8)
    
    # Add receiver labels
    for name, x, y in real_rx_pos:
        plt.text(x, y, name, fontsize=12, fontweight='bold', 
                ha='center', va='center', color='white', zorder=4)
    
    # Formatting
    plt.grid(True, alpha=0.4, linestyle='--', linewidth=1)
    plt.xlabel('X Position (meters)', fontsize=14, fontweight='bold')
    plt.ylabel('Y Position (meters)', fontsize=14, fontweight='bold')
    plt.title('Receiver Grid Layout with Inter-Node Distances', fontsize=16, fontweight='bold')
    
    
    plt.xlim(-2, x_max + 2)
    plt.ylim(-2, y_max + 2)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()

def main():
    # enter position in meters
    x_max = 50
    y_max = 50
    num_points = 1024
    
    goal_receiver_positions = [
        ('A', 24, 40),
        ('B', 10, 10),
        ('C', 40, 10),
        ('D', 25, 25),
        ('E', 35, 45)
    ]
    
    real_rx_pos = create_receiver_grid(x_max, y_max, num_points, goal_receiver_positions)
    
    print(f"Receiver Grid Configuration")
    print(f"=" * 80)
    print(f"Grid dimensions: {x_max}m x {y_max}m")
    print(f"Number of receivers: {len(real_rx_pos)}")
    print(f"Grid resolution: {num_points} points")
    print(f"Transmit power: {TX_POWER} dBm")
    print(f"Frequency: {FREQUENCY/1e9} GHz")
    print(f"Noise floor: {NOISE_FLOOR} dBm")
    print(f"=" * 80)
    print()
    
    # Print placed positions
    print("Placed Receiver Positions:")
    print(f"{'Name':<10} {'X (m)':<15} {'Y (m)':<15}")
    print("-" * 40)
    for name, x, y in real_rx_pos:
        print(f"{name:<10} {x:<15.2f} {y:<15.2f}")
    print()
    
    r_matrix = get_r_matrix(real_rx_pos)

    print("Distance Matrix")
    print(r_matrix)

    # Plot the grid
    plot_receiver_grid(real_rx_pos, x_max, y_max)


    

    return True

if __name__ == "__main__":
    main()