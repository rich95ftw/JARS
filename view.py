import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import stats

from controller import SimulationController


class JarsGUI(tk.Tk):
    """
    Main application window for the Jamming Analysis and Radio Simulation (JARS).
    """

    def __init__(self):
        """Initializes the main window and application components."""
        super().__init__()
        self.title("JARS - Jamming Analysis and Radio Simulation")
        self.controller = SimulationController()
        self.vars = self._init_variables()
        self._create_widgets()

    def _init_variables(self):
        """Initializes all Tkinter variables used for user inputs."""
        return {
            'tx_power': tk.DoubleVar(value=30.0),
            'tx_freq': tk.DoubleVar(value=300.0),
            'tx_x': tk.DoubleVar(value=0.0),
            'tx_y': tk.DoubleVar(value=0.0),
            'tx_z': tk.DoubleVar(value=0.0),

            'jammer_power': tk.DoubleVar(value=40.0),
            'jammer_freq': tk.DoubleVar(value=300.0),
            'jam_x': tk.DoubleVar(value=1000.0),
            'jam_y': tk.DoubleVar(value=500.0),
            'jam_z': tk.DoubleVar(value=0.0),

            'jammer_power_std': tk.DoubleVar(value=2.0),
            'jammer_x_std': tk.DoubleVar(value=100.0),
            'jammer_y_std': tk.DoubleVar(value=50.0),
            'jammer_z_std': tk.DoubleVar(value=20.0),

            'rx_sens': tk.DoubleVar(value=-90.0),
            'rx_x': tk.DoubleVar(value=2000.0),
            'rx_y': tk.DoubleVar(value=0.0),
            'rx_z': tk.DoubleVar(value=0.0),

            'threshold': tk.DoubleVar(value=10.0),
            'mc_samples': tk.IntVar(value=1000)
        }

    def _create_widgets(self):
        """Creates and packs all widgets in the main window."""
        input_frame = ttk.Frame(self, padding="10")
        input_frame.pack(fill=tk.X)

        self._create_input_frames(input_frame)
        self._create_button_frame()
        self._create_result_label()

    def _create_input_frames(self, parent_frame: ttk.Frame):
        """Adds all input sections to the main input frame."""
        self._add_input_section(parent_frame, "Transmitter", [
            ("Power (dBm)", 'tx_power'),
            ("Frequency (MHz)", 'tx_freq'),
            ("Position X (m)", 'tx_x'),
            ("Position Y (m)", 'tx_y'),
            ("Position Z (m)", 'tx_z'),
        ], column=0)

        self._add_input_section(parent_frame, "Jammer", [
            ("Power (dBm)", 'jammer_power'),
            ("Power Std (dB)", 'jammer_power_std'),
            ("Frequency (MHz)", 'jammer_freq'),
            ("Position X (m)", 'jam_x'),
            ("X Std (m)", 'jammer_x_std'),
            ("Position Y (m)", 'jam_y'),
            ("Y Std (m)", 'jammer_y_std'),
            ("Position Z (m)", 'jam_z'),
            ("Z Std (m)", 'jammer_z_std'),
        ], column=1)

        self._add_input_section(parent_frame, "Receiver & Threshold", [
            ("Sensitivity (dBm)", 'rx_sens'),
            ("Position X (m)", 'rx_x'),
            ("Position Y (m)", 'rx_y'),
            ("Position Z (m)", 'rx_z'),
            ("J/S Threshold (dB)", 'threshold'),
        ], column=2)

    def _create_button_frame(self):
        """Creates the frame containing all control buttons."""
        button_frame = ttk.Frame(self, padding="10")
        button_frame.pack(fill=tk.X)

        ttk.Button(
            button_frame, text="Run", command=self.run_simulation
        ).pack(side=tk.LEFT, expand=True, padx=5)
        ttk.Button(
            button_frame, text="Plot Geometry", command=self.plot_geometry
        ).pack(side=tk.LEFT, expand=True, padx=5)
        ttk.Button(
            button_frame, text="Monte Carlo",
            command=self.run_monte_carlo_sim
        ).pack(side=tk.LEFT, expand=True, padx=5)

        ttk.Label(button_frame, text="Samples:").pack(side=tk.LEFT)
        ttk.Entry(button_frame,
                  textvariable=self.vars['mc_samples'],
                  width=6).pack(side=tk.LEFT)

    def _create_result_label(self):
        """Creates the label for displaying simulation results."""
        self.result_label = ttk.Label(
            self,
            text="Press 'Run Simulation' to see results.",
            padding="10",
            font=("", 10, "bold")
        )
        self.result_label.pack(fill=tk.X)

    def _add_input_section(self, parent, title, fields, column):
        """
        Helper method to create a labeled frame with entry fields.
        """
        frame = ttk.LabelFrame(parent, text=title)
        frame.grid(row=0, column=column, padx=10, pady=5, sticky=(tk.W, tk.E))
        for i, (label_text, var_name) in enumerate(fields):
            ttk.Label(frame, text=label_text).grid(
                row=i, column=0, sticky=tk.W, padx=5, pady=2
            )
            ttk.Entry(frame,
                      textvariable=self.vars[var_name],
                      width=10).grid(
                row=i, column=1, sticky=(tk.W, tk.E), padx=5, pady=2
            )

    def run_simulation(self):
        """
        Runs a single deterministic simulation and updates the result label.
        """
        try:
            tx = self.controller.create_radio_source(
                self.vars['tx_power'].get(),
                self.vars['tx_freq'].get(),
                self.vars['tx_x'].get(),
                self.vars['tx_y'].get(),
                self.vars['tx_z'].get()
            )
            jammer = self.controller.create_radio_source(
                self.vars['jammer_power'].get(),
                self.vars['jammer_freq'].get(),
                self.vars['jam_x'].get(),
                self.vars['jam_y'].get(),
                self.vars['jam_z'].get()
            )
            rx = self.controller.create_receiver(
                self.vars['rx_sens'].get(),
                self.vars['rx_x'].get(),
                self.vars['rx_y'].get(),
                self.vars['rx_z'].get()
            )
            threshold = self.vars['threshold'].get()

            results = self.controller.run_simulation(tx, jammer, rx, threshold)
            j_s = results["j_s_db"]
            comm_success = results["communication_success"]

            result_text = f"J/S ratio: {j_s:.2f} dB\n"
            if comm_success:
                result_text += "Communication is likely SUCCESSFUL."
            else:
                result_text += "Communication is likely BLOCKED."

            self.result_label.config(text=result_text)
        except Exception as e:
            # Consider catching more specific exceptions if possible
            self.result_label.config(text=f"Error: {e}")

    def run_monte_carlo_sim(self):
        """
        Runs a Monte Carlo simulation and displays results in a new window.
        """
        try:
            tx_power = self.vars['tx_power'].get()
            tx_freq = self.vars['tx_freq'].get()
            tx_pos = (self.vars['tx_x'].get(), self.vars['tx_y'].get(),
                      self.vars['tx_z'].get())

            rx_sens = self.vars['rx_sens'].get()
            rx_pos = (self.vars['rx_x'].get(), self.vars['rx_y'].get(),
                      self.vars['rx_z'].get())

            jam_power_mean = self.vars['jammer_power'].get()
            jam_power_std = self.vars['jammer_power_std'].get()
            jam_freq = self.vars['jammer_freq'].get()

            jam_x = self.vars['jam_x'].get()
            jam_x_std = self.vars['jammer_x_std'].get()
            jam_y = self.vars['jam_y'].get()
            jam_y_std = self.vars['jammer_y_std'].get()
            jam_z = self.vars['jam_z'].get()
            jam_z_std = self.vars['jammer_z_std'].get()

            jam_x_dist = stats.norm(loc=jam_x, scale=jam_x_std)
            jam_y_dist = stats.norm(loc=jam_y, scale=jam_y_std)
            jam_z_dist = stats.norm(loc=jam_z, scale=jam_z_std)

            N = self.vars['mc_samples'].get()
            result = self.controller.run_monte_carlo(
                tx_power, tx_freq, tx_pos,
                rx_sens, rx_pos,
                jam_power_mean, jam_power_std,
                jam_freq, jam_x_dist, jam_y_dist, jam_z_dist,
                N=N
            )

            fig, ax = plt.subplots()
            ax.hist(result['js_array'], bins=50, alpha=0.75)
            ax.set_title("Monte Carlo J/S Ratio Distribution")
            ax.set_xlabel("J/S (dB)")
            ax.set_ylabel("Frequency")
            ax.grid(True)

            top = tk.Toplevel(self)
            top.title("Monte Carlo Results")
            canvas = FigureCanvasTkAgg(fig, master=top)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            self.result_label.config(text=(
                f"Monte Carlo Simulation Complete\n"
                f"Mean J/S: {result['mean_js']:.2f} dB\n"
                f"50th percentile: {result['percentile_50']:.2f} dB\n"
                f"90th percentile: {result['percentile_90']:.2f} dB"
            ))

        except Exception as e:
            self.result_label.config(text=f"Monte Carlo Error: {e}")

    def plot_geometry(self):
        """
        Generates and displays a plot of the geometry in a new window.
        """
        try:
            tx = self.controller.create_radio_source(
                self.vars['tx_power'].get(), self.vars['tx_freq'].get(),
                self.vars['tx_x'].get(), self.vars['tx_y'].get(),
                self.vars['tx_z'].get()
            )
            jammer = self.controller.create_radio_source(
                self.vars['jammer_power'].get(),
                self.vars['jammer_freq'].get(),
                self.vars['jam_x'].get(), self.vars['jam_y'].get(),
                self.vars['jam_z'].get()
            )
            rx = self.controller.create_receiver(
                self.vars['rx_sens'].get(),
                self.vars['rx_x'].get(), self.vars['rx_y'].get(),
                self.vars['rx_z'].get()
            )
            threshold = self.vars['threshold'].get()

            results = self.controller.run_simulation(tx, jammer, rx,
                                                    threshold)
            tx_pos = tx.position
            jam_pos = jammer.position
            rx_pos = rx.position

            fig, ax = plt.subplots(figsize=(8, 7))
            ax.set_title("JARS Geometry Plot")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")

            ax.scatter(tx_pos.x, tx_pos.y, color='blue',
                       label='Transmitter', s=100)
            ax.scatter(jam_pos.x, jam_pos.y, color='red', label='Jammer', s=100)
            ax.scatter(rx_pos.x, rx_pos.y, color='green',
                       label='Receiver', s=100)

            ax.plot([tx_pos.x, rx_pos.x], [tx_pos.y, rx_pos.y], 'b--')
            ax.plot([jam_pos.x, rx_pos.x], [jam_pos.y, rx_pos.y], 'r--')

            tx_recv = results["tx_recv_dbm"]
            jam_recv = results["jam_recv_dbm"]
            j_s = results["j_s_db"]
            comm_success = results["communication_success"]

            tx_dist = tx_pos.distance_to(rx_pos)
            jam_dist = jam_pos.distance_to(rx_pos)

            ax.annotate(f"Tx → Rx\n{tx_dist:.1f} m\n{tx_recv:.1f} dBm",
                        xy=((tx_pos.x + rx_pos.x) / 2, (tx_pos.y + rx_pos.y) / 2),
                        ha='center', va='center', color='blue',
                        bbox=dict(boxstyle="round", fc="white", alpha=0.7))

            ax.annotate(f"Jam → Rx\n{jam_dist:.1f} m\n{jam_recv:.1f} dBm",
                        xy=((jam_pos.x + rx_pos.x) / 2, (jam_pos.y + rx_pos.y) / 2),
                        ha='center', va='center', color='red',
                        bbox=dict(boxstyle="round", fc="white", alpha=0.7))

            ax.text(0.05, 0.95, f"J/S = {j_s:.1f} dB",
                    transform=ax.transAxes, fontsize=8, fontweight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            status_text = "Communication success" if comm_success else \
                          "Communication blocked"
            status_color = 'lightgreen' if comm_success else '#ffcccb'

            ax.text(0.05, 0.87, status_text,
                    transform=ax.transAxes, fontsize=8, fontweight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor=status_color,
                              alpha=0.8))

            ax.legend()
            ax.grid(True)
            ax.set_aspect('equal', adjustable='box')
            plt.tight_layout()

            top = tk.Toplevel(self)
            top.title("Geometry Plot")
            canvas = FigureCanvasTkAgg(fig, master=top)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            self.result_label.config(text=f"Plot error: {e}")


if __name__ == "__main__":
    app = JarsGUI()
    app.mainloop()