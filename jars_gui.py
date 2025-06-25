import tkinter as tk
from tkinter import ttk
from math import log10, sqrt

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Position:
    """A simple class to hold 3D coordinates."""
    def __init__(self, x, y, z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def distance_to(self, other):
        """Calculates the Euclidean distance to another Position object."""
        return sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)

class RadioSource:
    """Represents a radio transmitter or a jammer."""
    def __init__(self, power_dbm, frequency_mhz, position):
        self.power_dbm = float(power_dbm)
        self.frequency_mhz = float(frequency_mhz)
        self.position = position

class Receiver:
    """Represents a radio receiver."""
    def __init__(self, sensitivity_dbm, position):
        self.sensitivity_dbm = float(sensitivity_dbm)
        self.position = position

def fspl_db(distance_km, frequency_mhz):
    """Calculates Free Space Path Loss (FSPL) in dB."""
    if distance_km <= 0 or frequency_mhz <= 0:
        return float('inf')
    return 20 * log10(distance_km) + 20 * log10(frequency_mhz) + 32.44

def received_power_dbm(tx, rx_pos):
    """Calculates the received power at a position from a source."""
    distance_km = tx.position.distance_to(rx_pos) / 1000.0
    return tx.power_dbm - fspl_db(distance_km, tx.frequency_mhz)

def j_s_ratio_db(jammer, transmitter, receiver):
    """Calculates the Jammer-to-Signal (J/S) ratio in dB."""
    j_recv = received_power_dbm(jammer, receiver.position)
    s_recv = received_power_dbm(transmitter, receiver.position)
    return j_recv - s_recv

def is_jamming_successful(j_s_db, threshold_db):
    """Determines if jamming is successful based on a threshold."""
    return j_s_db > threshold_db

# --- GUI ---

def run_simulation():
    """Callback function to run the simulation and display results."""
    try:
        # Get user input from GUI variables
        tx = RadioSource(
            power_dbm=tx_power_var.get(),
            frequency_mhz=tx_freq_var.get(),
            position=Position(tx_x_var.get(), tx_y_var.get(), tx_z_var.get())
        )

        jammer = RadioSource(
            power_dbm=jammer_power_var.get(),
            frequency_mhz=jammer_freq_var.get(),
            position=Position(jam_x_var.get(), jam_y_var.get(), jam_z_var.get())
        )

        rx = Receiver(
            sensitivity_dbm=rx_sens_var.get(),
            position=Position(rx_x_var.get(), rx_y_var.get(), rx_z_var.get())
        )

        threshold = float(threshold_var.get())

        # Calculate results
        j_s = j_s_ratio_db(jammer, tx, rx)
        success = is_jamming_successful(j_s, threshold)

        # Update result label
        result_text = f"J/S ratio: {j_s:.2f} dB\n"
        result_text += "Jamming is likely SUCCESSFUL." if success else "Jamming is likely UNSUCCESSFUL."
        result_label.config(text=result_text)
    except Exception as e:
        result_label.config(text=f"Error: {e}")

def plot_geometry():
    """Callback function to create and display the geometry plot."""
    try:
        # Get user input
        tx = RadioSource(
            power_dbm=tx_power_var.get(),
            frequency_mhz=tx_freq_var.get(),
            position=Position(tx_x_var.get(), tx_y_var.get(), tx_z_var.get())
        )
        jammer = RadioSource(
            power_dbm=jammer_power_var.get(),
            frequency_mhz=jammer_freq_var.get(),
            position=Position(jam_x_var.get(), jam_y_var.get(), jam_z_var.get())
        )
        rx = Receiver(
            sensitivity_dbm=rx_sens_var.get(),
            position=Position(rx_x_var.get(), rx_y_var.get(), rx_z_var.get())
        )

        # Get positions for plotting
        tx_pos = tx.position
        jam_pos = jammer.position
        rx_pos = rx.position

        # Calculate distances and received powers for annotations
        tx_dist = tx_pos.distance_to(rx_pos)
        jam_dist = jam_pos.distance_to(rx_pos)
        tx_recv = received_power_dbm(tx, rx_pos)
        jam_recv = received_power_dbm(jammer, rx_pos)
        j_s = jam_recv - tx_recv

        # Create Matplotlib plot
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.set_title("JARS Geometry Plot")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

        # Plot Transmitter, Jammer, and Receiver locations
        ax.scatter(tx_pos.x, tx_pos.y, color='blue', label='Transmitter', s=100, zorder=5)
        ax.scatter(jam_pos.x, jam_pos.y, color='red', label='Jammer', s=100, zorder=5)
        ax.scatter(rx_pos.x, rx_pos.y, color='green', label='Receiver', s=100, zorder=5)

        # Plot lines connecting sources to the receiver
        ax.plot([tx_pos.x, rx_pos.x], [tx_pos.y, rx_pos.y], 'b--', zorder=1)
        ax.plot([jam_pos.x, rx_pos.x], [jam_pos.y, rx_pos.y], 'r--', zorder=1)

        # Annotate lines with distance and received power
        ax.annotate(f"Tx → Rx\n{tx_dist:.1f} m\n{tx_recv:.1f} dBm",
                    xy=((tx_pos.x + rx_pos.x)/2, (tx_pos.y + rx_pos.y)/2),
                    ha='center', va='bottom', color='blue',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7))

        ax.annotate(f"Jam → Rx\n{jam_dist:.1f} m\n{jam_recv:.1f} dBm",
                    xy=((jam_pos.x + rx_pos.x)/2, (jam_pos.y + rx_pos.y)/2),
                    ha='center', va='bottom', color='red',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7))

        # --- REFACTORED: Non-overlapping Annotations ---
        # Place annotations in a fixed position on the plot (top-left corner)
        # to prevent them from overlapping with the plot data.
        # We use `ax.transAxes` for positioning relative to the axes dimensions,
        # where (0,0) is the bottom-left and (1,1) is the top-right.

        # 1. J/S Ratio Annotation
        ax.text(0.05, 0.95, f"J/S = {j_s:.1f} dB",
                transform=ax.transAxes,
                fontsize=10,
                fontweight='bold',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # 2. Jamming Success/Failure Annotation
        threshold = float(threshold_var.get())
        jamming_successful = is_jamming_successful(j_s, threshold)
        status_text = "Jamming success" if jamming_successful else "Jamming failure"
        # Use light green for success and light red for failure
        status_color = 'lightgreen' if jamming_successful else '#ffcccb'

        ax.text(0.05, 0.85, status_text,
                transform=ax.transAxes,
                fontsize=10,
                fontweight='bold',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.8))


        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()


        # Embed plot in a new Tkinter Toplevel window
        top = tk.Toplevel(root)
        top.title("Geometry Plot")
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    except Exception as e:
        result_label.config(text=f"Plot error: {e}")

# --- Main GUI Window Setup ---
root = tk.Tk()
root.title("JARS - Jamming Analysis and Reporting System")

# Frame for inputs
input_frame = ttk.Frame(root, padding="10")
input_frame.pack(fill=tk.X)

# Define and initialize Tkinter variables for user input
tx_power_var = tk.DoubleVar(value=30.0)
tx_freq_var = tk.DoubleVar(value=300.0)
tx_x_var, tx_y_var, tx_z_var = tk.DoubleVar(value=0), tk.DoubleVar(value=0), tk.DoubleVar(value=0)

jammer_power_var = tk.DoubleVar(value=40.0)
jammer_freq_var = tk.DoubleVar(value=300.0)
jam_x_var, jam_y_var, jam_z_var = tk.DoubleVar(value=1000), tk.DoubleVar(value=500), tk.DoubleVar(value=0)

rx_sens_var = tk.DoubleVar(value=-90.0)
rx_x_var, rx_y_var, rx_z_var = tk.DoubleVar(value=2000), tk.DoubleVar(value=0), tk.DoubleVar(value=0)

threshold_var = tk.DoubleVar(value=10.0)

# Create labeled entry widgets
def make_labeled_entry(parent, label_text, var, row, col):
    """Helper function to create a label and an entry widget."""
    ttk.Label(parent, text=label_text).grid(row=row, column=col, sticky=tk.W, padx=5, pady=2)
    ttk.Entry(parent, textvariable=var, width=10).grid(row=row, column=col + 1, sticky=(tk.W, tk.E), padx=5, pady=2)

# --- Layout of Input Fields using a grid ---

# Transmitter Inputs
tx_frame = ttk.LabelFrame(input_frame, text="Transmitter")
tx_frame.grid(row=0, column=0, padx=10, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
make_labeled_entry(tx_frame, "Power (dBm)", tx_power_var, 0, 0)
make_labeled_entry(tx_frame, "Frequency (MHz)", tx_freq_var, 1, 0)
make_labeled_entry(tx_frame, "Position X (m)", tx_x_var, 2, 0)
make_labeled_entry(tx_frame, "Position Y (m)", tx_y_var, 3, 0)
make_labeled_entry(tx_frame, "Position Z (m)", tx_z_var, 4, 0)


# Jammer Inputs
jammer_frame = ttk.LabelFrame(input_frame, text="Jammer")
jammer_frame.grid(row=0, column=1, padx=10, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
make_labeled_entry(jammer_frame, "Power (dBm)", jammer_power_var, 0, 0)
make_labeled_entry(jammer_frame, "Frequency (MHz)", jammer_freq_var, 1, 0)
make_labeled_entry(jammer_frame, "Position X (m)", jam_x_var, 2, 0)
make_labeled_entry(jammer_frame, "Position Y (m)", jam_y_var, 3, 0)
make_labeled_entry(jammer_frame, "Position Z (m)", jam_z_var, 4, 0)

# Receiver and Threshold Inputs
rx_frame = ttk.LabelFrame(input_frame, text="Receiver & Threshold")
rx_frame.grid(row=0, column=2, padx=10, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
make_labeled_entry(rx_frame, "Sensitivity (dBm)", rx_sens_var, 0, 0)
make_labeled_entry(rx_frame, "Position X (m)", rx_x_var, 1, 0)
make_labeled_entry(rx_frame, "Position Y (m)", rx_y_var, 2, 0)
make_labeled_entry(rx_frame, "Position Z (m)", rx_z_var, 3, 0)
make_labeled_entry(rx_frame, "J/S Threshold (dB)", threshold_var, 4, 0)

# --- Buttons and Result Label ---
button_frame = ttk.Frame(root, padding="10")
button_frame.pack(fill=tk.X)

ttk.Button(button_frame, text="Run Simulation", command=run_simulation).pack(side=tk.LEFT, expand=True, padx=5)
ttk.Button(button_frame, text="Plot Geometry", command=plot_geometry).pack(side=tk.LEFT, expand=True, padx=5)

result_label = ttk.Label(root, text="Press 'Run Simulation' to see results.", padding="10", font=("", 10, "bold"))
result_label.pack(fill=tk.X)

# Start the Tkinter event loop
root.mainloop()
