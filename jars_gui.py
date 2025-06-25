import tkinter as tk
from tkinter import ttk
from math import log10, sqrt

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Position:
    def __init__(self, x, y, z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def distance_to(self, other):
        return sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)

class RadioSource:
    def __init__(self, power_dbm, frequency_mhz, position):
        self.power_dbm = float(power_dbm)
        self.frequency_mhz = float(frequency_mhz)
        self.position = position

class Receiver:
    def __init__(self, sensitivity_dbm, position):
        self.sensitivity_dbm = float(sensitivity_dbm)
        self.position = position

def fspl_db(distance_km, frequency_mhz):
    return 20 * log10(distance_km) + 20 * log10(frequency_mhz) + 32.44

def received_power_dbm(tx, rx_pos):
    distance_km = tx.position.distance_to(rx_pos) / 1000.0
    return tx.power_dbm - fspl_db(distance_km, tx.frequency_mhz)

def j_s_ratio_db(jammer, transmitter, receiver):
    j_recv = received_power_dbm(jammer, receiver.position)
    s_recv = received_power_dbm(transmitter, receiver.position)
    return j_recv - s_recv

def is_jamming_successful(j_s_db, threshold_db):
    return j_s_db > threshold_db

# GUI
def run_simulation():
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

        threshold = float(threshold_var.get())

        j_s = j_s_ratio_db(jammer, tx, rx)
        result = f"J/S ratio: {j_s:.2f} dB\n"
        result += "Jamming is likely SUCCESSFUL." if is_jamming_successful(j_s, threshold) else "Jamming is likely UNSUCCESSFUL."
        result_label.config(text=result)
    except Exception as e:
        result_label.config(text=f"Error: {e}")

# Main GUI window
root = tk.Tk()
root.title("JARS - Jamming Analysis Tool")

def make_input(label, var):
    ttk.Label(root, text=label).pack()
    return ttk.Entry(root, textvariable=var)

# Define variables
tx_power_var = tk.DoubleVar(value=30.0)
tx_freq_var = tk.DoubleVar(value=300.0)
tx_x_var, tx_y_var, tx_z_var = tk.DoubleVar(value=0), tk.DoubleVar(value=0), tk.DoubleVar(value=0)

jammer_power_var = tk.DoubleVar(value=40.0)
jammer_freq_var = tk.DoubleVar(value=300.0)
jam_x_var, jam_y_var, jam_z_var = tk.DoubleVar(value=1000), tk.DoubleVar(value=0), tk.DoubleVar(value=0)

rx_sens_var = tk.DoubleVar(value=-90.0)
rx_x_var, rx_y_var, rx_z_var = tk.DoubleVar(value=500), tk.DoubleVar(value=0), tk.DoubleVar(value=0)

threshold_var = tk.DoubleVar(value=10.0)

# Layout
for lbl, var in [
    ("Transmitter Power (dBm)", tx_power_var),
    ("Transmitter Frequency (MHz)", tx_freq_var),
    ("Transmitter X", tx_x_var), ("Transmitter Y", tx_y_var), ("Transmitter Z", tx_z_var),
    ("Jammer Power (dBm)", jammer_power_var),
    ("Jammer Frequency (MHz)", jammer_freq_var),
    ("Jammer X", jam_x_var), ("Jammer Y", jam_y_var), ("Jammer Z", jam_z_var),
    ("Receiver Sensitivity (dBm)", rx_sens_var),
    ("Receiver X", rx_x_var), ("Receiver Y", rx_y_var), ("Receiver Z", rx_z_var),
    ("Jamming Success Threshold (dB)", threshold_var)
]:
    make_input(lbl, var).pack()

def plot_geometry():
    try:
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

        # Geometry
        tx_pos = tx.position
        jam_pos = jammer.position
        rx_pos = rx.position

        tx_dist = tx_pos.distance_to(rx_pos)
        jam_dist = jam_pos.distance_to(rx_pos)
        tx_recv = received_power_dbm(tx, rx_pos)
        jam_recv = received_power_dbm(jammer, rx_pos)
        j_s = jam_recv - tx_recv

        # Plot
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_title("JARS Geometry Plot")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

        ax.scatter(tx_pos.x, tx_pos.y, color='blue', label='Transmitter')
        ax.scatter(jam_pos.x, jam_pos.y, color='red', label='Jammer')
        ax.scatter(rx_pos.x, rx_pos.y, color='green', label='Receiver')

        ax.plot([tx_pos.x, rx_pos.x], [tx_pos.y, rx_pos.y], 'b--')
        ax.plot([jam_pos.x, rx_pos.x], [jam_pos.y, rx_pos.y], 'r--')

        # Annotations
        ax.annotate(f"Tx → Rx\n{tx_dist:.1f} m\n{tx_recv:.1f} dBm",
                    xy=((tx_pos.x + rx_pos.x)/2, (tx_pos.y + rx_pos.y)/2),
                    ha='center', va='bottom', color='blue')

        ax.annotate(f"Jam → Rx\n{jam_dist:.1f} m\n{jam_recv:.1f} dBm",
                    xy=((jam_pos.x + rx_pos.x)/2, (jam_pos.y + rx_pos.y)/2),
                    ha='center', va='bottom', color='red')

        # Annotation offset
        offset_x = 30
        offset_y = 30

        # Compute limits from data
        x_vals = [tx_pos.x, jam_pos.x, rx_pos.x]
        y_vals = [tx_pos.y, jam_pos.y, rx_pos.y]
        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)

        # Clamp annotation position
        anno_x = min(max(rx_pos.x + offset_x, x_min), x_max)
        anno_y = min(max(rx_pos.y + offset_y, y_min), y_max)

        ax.annotate(f"J/S = {j_s:.1f} dB",
                    xy=(rx_pos.x, rx_pos.y), xytext=(anno_x, anno_y),
                    bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))

        ax.legend()
        ax.grid(True)

        # Embed in Tkinter
        top = tk.Toplevel(root)
        top.title("Geometry Plot")
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    except Exception as e:
        result_label.config(text=f"Plot error: {e}")

ttk.Button(root, text="Run Simulation", command=run_simulation).pack(pady=10)
ttk.Button(root, text="Plot Geometry", command=plot_geometry).pack(pady=5)
result_label = ttk.Label(root, text="")
result_label.pack()

root.mainloop()
