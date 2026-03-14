import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import stats

from jars.controller import SimulationController


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
            'jammer_pos_dist': tk.StringVar(value="Normal"),

            'rx_sens': tk.DoubleVar(value=-90.0),
            'rx_x': tk.DoubleVar(value=2000.0),
            'rx_x_std': tk.DoubleVar(value=0.0),
            'rx_y': tk.DoubleVar(value=0.0),
            'rx_y_std': tk.DoubleVar(value=0.0),
            'rx_z': tk.DoubleVar(value=0.0),
            'rx_z_std': tk.DoubleVar(value=0.0),

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

        self._create_jammer_section(parent_frame, column=1)

        self._add_input_section(parent_frame, "Receiver & Threshold", [
            ("Sensitivity (dBm)", 'rx_sens'),
            ("Position X (m)", 'rx_x'),
            ("X Std (m)", 'rx_x_std'),
            ("Position Y (m)", 'rx_y'),
            ("Y Std (m)", 'rx_y_std'),
            ("Position Z (m)", 'rx_z'),
            ("Z Std (m)", 'rx_z_std'),
            ("J/S Threshold (dB)", 'threshold'),
        ], column=2)

    def _create_jammer_section(self, parent: ttk.Frame, column: int):
        """
        Creates the Jammer input section with a position distribution selector.
        """
        frame = ttk.LabelFrame(parent, text="Jammer")
        frame.grid(row=0, column=column, padx=10, pady=5, sticky=(tk.W, tk.E))

        for i, (label_text, var_name) in enumerate([
            ("Power (dBm)", 'jammer_power'),
            ("Power Std (dB)", 'jammer_power_std'),
            ("Frequency (MHz)", 'jammer_freq'),
        ]):
            ttk.Label(frame, text=label_text).grid(
                row=i, column=0, sticky=tk.W, padx=5, pady=2)
            ttk.Entry(frame, textvariable=self.vars[var_name], width=10).grid(
                row=i, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)

        ttk.Label(frame, text="Pos. Distribution").grid(
            row=3, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Combobox(
            frame,
            textvariable=self.vars['jammer_pos_dist'],
            values=["Normal", "Uniform"],
            state="readonly",
            width=8,
        ).grid(row=3, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)

        for i, (label_text, var_name) in enumerate([
            ("Position X (m)", 'jam_x'),
            ("X Spread (m)", 'jammer_x_std'),
            ("Position Y (m)", 'jam_y'),
            ("Y Spread (m)", 'jammer_y_std'),
            ("Position Z (m)", 'jam_z'),
            ("Z Spread (m)", 'jammer_z_std'),
        ], start=4):
            ttk.Label(frame, text=label_text).grid(
                row=i, column=0, sticky=tk.W, padx=5, pady=2)
            ttk.Entry(frame, textvariable=self.vars[var_name], width=10).grid(
                row=i, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)

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
            text="Press 'Run' to see results.",
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
            jam_success = results["jamming_success"]

            result_text = f"J/S ratio: {j_s:.2f} dB\n"
            if comm_success:
                result_text += "Communication is likely SUCCESSFUL."
            elif jam_success:
                result_text += "Communication is likely BLOCKED (jammed)."
            else:
                result_text += "Communication is likely BLOCKED (signal too weak)."

            if results["frequency_mismatch"]:
                result_text += (
                    f"\n\u26a0 WARNING: Jammer frequency "
                    f"({jammer.frequency_mhz:.1f} MHz) \u2260 "
                    f"TX frequency ({tx.frequency_mhz:.1f} MHz). "
                    f"J/S result may not be valid for spot jamming."
                )

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
            rx_x_std = self.vars['rx_x_std'].get()
            rx_y_std = self.vars['rx_y_std'].get()
            rx_z_std = self.vars['rx_z_std'].get()

            jam_power_mean = self.vars['jammer_power'].get()
            jam_power_std = self.vars['jammer_power_std'].get()
            jam_freq = self.vars['jammer_freq'].get()

            jam_x = self.vars['jam_x'].get()
            jam_x_std = self.vars['jammer_x_std'].get()
            jam_y = self.vars['jam_y'].get()
            jam_y_std = self.vars['jammer_y_std'].get()
            jam_z = self.vars['jam_z'].get()
            jam_z_std = self.vars['jammer_z_std'].get()

            dist_type = self.vars['jammer_pos_dist'].get()

            def _make_pos_dist(centre, spread):
                if spread <= 0.0:
                    return stats.norm(loc=centre, scale=1e-6)
                if dist_type == "Uniform":
                    return stats.uniform(loc=centre - spread,
                                        scale=2 * spread)
                return stats.norm(loc=centre, scale=spread)

            jam_x_dist = _make_pos_dist(jam_x, jam_x_std)
            jam_y_dist = _make_pos_dist(jam_y, jam_y_std)
            jam_z_dist = _make_pos_dist(jam_z, jam_z_std)

            threshold = self.vars['threshold'].get()
            N = self.vars['mc_samples'].get()
            result = self.controller.run_monte_carlo(
                tx_power, tx_freq, tx_pos,
                rx_sens, rx_pos,
                jam_power_mean, jam_power_std,
                jam_freq, jam_x_dist, jam_y_dist, jam_z_dist,
                j_s_threshold_db=threshold,
                N=N,
                rx_x_std=rx_x_std,
                rx_y_std=rx_y_std,
                rx_z_std=rx_z_std,
            )

            fig, ax = plt.subplots()
            ax.hist(result['js_array'], bins=min(50, max(10, N // 10)),
                    alpha=0.75)
            ax.axvline(threshold, color='red', linestyle='--',
                       label=f'J/S Threshold ({threshold:.1f} dB)')
            ax.set_title(
                f"Monte Carlo J/S Ratio Distribution"
                f"\nJammer position: {dist_type}"
            )
            ax.set_xlabel("J/S (dB)")
            ax.set_ylabel("Count")
            ax.legend()
            ax.grid(True)

            top = tk.Toplevel(self)
            top.title("Monte Carlo Results")
            canvas = FigureCanvasTkAgg(fig, master=top)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            tx_recv_label = (
                "Mean Tx signal at Rx" if result['rx_position_uncertain']
                else "Tx signal at Rx"
            )
            mc_text = (
                f"Monte Carlo Simulation Complete"
                f"  |  Jammer position: {dist_type}\n"
                f"{tx_recv_label}: {result['tx_recv_dbm']:.2f} dBm\n"
                f"Mean J/S: {result['mean_js']:.2f} dB  |  "
                f"50th pct: {result['percentile_50']:.2f} dB  |  "
                f"90th pct: {result['percentile_90']:.2f} dB\n"
                f"P(jamming success): {result['p_jamming_success']:.1%}  |  "
                f"P(comm success): {result['p_comm_success']:.1%}"
            )
            if result['frequency_mismatch']:
                mc_text += (
                    f"\n\u26a0 WARNING: Jammer frequency ({jam_freq:.1f} MHz)"
                    f" \u2260 TX frequency ({tx_freq:.1f} MHz). "
                    f"J/S result may not be valid for spot jamming."
                )
            self.result_label.config(text=mc_text)

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
            jam_success = results["jamming_success"]

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

            if comm_success:
                status_text = "Communication success"
                status_color = 'lightgreen'
            elif jam_success:
                status_text = "Communication blocked (jammed)"
                status_color = '#ffcccb'
            else:
                status_text = "Communication blocked (signal too weak)"
                status_color = '#ffcccb'

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


def main():
    app = JarsGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
