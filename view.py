import tkinter as tk
from tkinter import ttk
from controller import SimulationController
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


class JarsGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("JARS - Jamming Analysis and Radio Simulation")

        self.controller = SimulationController()
        self._create_widgets()

    def _create_widgets(self):
        self.vars = {
            'tx_power': tk.DoubleVar(value=30.0),
            'tx_freq': tk.DoubleVar(value=300.0),
            'tx_x': tk.DoubleVar(value=0),
            'tx_y': tk.DoubleVar(value=0),
            'tx_z': tk.DoubleVar(value=0),

            'jam_power': tk.DoubleVar(value=40.0),
            'jam_freq': tk.DoubleVar(value=300.0),
            'jam_x': tk.DoubleVar(value=1000),
            'jam_y': tk.DoubleVar(value=500),
            'jam_z': tk.DoubleVar(value=0),

            'rx_sens': tk.DoubleVar(value=-90.0),
            'rx_x': tk.DoubleVar(value=2000),
            'rx_y': tk.DoubleVar(value=0),
            'rx_z': tk.DoubleVar(value=0),

            'threshold': tk.DoubleVar(value=10.0)
        }

        input_frame = ttk.Frame(self, padding="10")
        input_frame.pack(fill=tk.X)

        self._add_input_section(input_frame, "Transmitter", ['tx_power', 'tx_freq', 'tx_x', 'tx_y', 'tx_z'], 0)
        self._add_input_section(input_frame, "Jammer", ['jam_power', 'jam_freq', 'jam_x', 'jam_y', 'jam_z'], 1)
        self._add_input_section(input_frame, "Receiver & Threshold", ['rx_sens', 'rx_x', 'rx_y', 'rx_z', 'threshold'], 2)

        button_frame = ttk.Frame(self, padding="10")
        button_frame.pack(fill=tk.X)

        ttk.Button(button_frame, text="Run Simulation", command=self.run_simulation).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Plot Geometry", command=self.plot_geometry).pack(side=tk.LEFT, padx=5)

        self.result_label = ttk.Label(self, text="Press 'Run Simulation' to see results.", padding="10", font=('', 10, 'bold'))
        self.result_label.pack(fill=tk.X)

    def _add_input_section(self, parent, title, fields, col):
        frame = ttk.LabelFrame(parent, text=title)
        frame.grid(row=0, column=col, padx=10, pady=5, sticky=tk.N)
        for i, field in enumerate(fields):
            label = field.replace('_', ' ').title()
            ttk.Label(frame, text=label).grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
            ttk.Entry(frame, textvariable=self.vars[field], width=10).grid(row=i, column=1, padx=5, pady=2)

    def run_simulation(self):
        try:
            tx = self.controller.create_radio_source(
                self.vars['tx_power'].get(), self.vars['tx_freq'].get(),
                self.vars['tx_x'].get(), self.vars['tx_y'].get(), self.vars['tx_z'].get()
            )
            jammer = self.controller.create_radio_source(
                self.vars['jam_power'].get(), self.vars['jam_freq'].get(),
                self.vars['jam_x'].get(), self.vars['jam_y'].get(), self.vars['jam_z'].get()
            )
            rx = self.controller.create_receiver(
                self.vars['rx_sens'].get(),
                self.vars['rx_x'].get(), self.vars['rx_y'].get(), self.vars['rx_z'].get()
            )
            threshold = self.vars['threshold'].get()

            j_s, success = self.controller.run_simulation(tx, jammer, rx, threshold)

            result = f"J/S ratio: {j_s:.2f} dB\n"
            result += "Jamming is likely SUCCESSFUL." if success else "Jamming is likely UNSUCCESSFUL."
            self.result_label.config(text=result)
        except Exception as e:
            self.result_label.config(text=f"Error: {e}")

    def plot_geometry(self):
        try:
            tx = self.controller.create_radio_source(
                self.vars['tx_power'].get(), self.vars['tx_freq'].get(),
                self.vars['tx_x'].get(), self.vars['tx_y'].get(), self.vars['tx_z'].get()
            )
            jammer = self.controller.create_radio_source(
                self.vars['jam_power'].get(), self.vars['jam_freq'].get(),
                self.vars['jam_x'].get(), self.vars['jam_y'].get(), self.vars['jam_z'].get()
            )
            rx = self.controller.create_receiver(
                self.vars['rx_sens'].get(),
                self.vars['rx_x'].get(), self.vars['rx_y'].get(), self.vars['rx_z'].get()
            )
            threshold = self.vars['threshold'].get()

            tx_recv, jam_recv = self.controller.get_received_powers(tx, jammer, rx)
            j_s = jam_recv - tx_recv
            success = self.controller.run_simulation(tx, jammer, rx, threshold)[1]

            fig, ax = plt.subplots(figsize=(8, 7))
            ax.set_title("JARS Geometry Plot")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")

            ax.scatter(tx.position.x, tx.position.y, color='blue', label='Transmitter', s=100)
            ax.scatter(jammer.position.x, jammer.position.y, color='red', label='Jammer', s=100)
            ax.scatter(rx.position.x, rx.position.y, color='green', label='Receiver', s=100)

            ax.plot([tx.position.x, rx.position.x], [tx.position.y, rx.position.y], 'b--')
            ax.plot([jammer.position.x, rx.position.x], [jammer.position.y, rx.position.y], 'r--')

            ax.annotate(f"Tx → Rx\n{tx_recv:.1f} dBm",
                        xy=((tx.position.x + rx.position.x)/2, (tx.position.y + rx.position.y)/2),
                        ha='center', va='center', color='blue')

            ax.annotate(f"Jam → Rx\n{jam_recv:.1f} dBm",
                        xy=((jammer.position.x + rx.position.x)/2, (jammer.position.y + rx.position.y)/2),
                        ha='center', va='center', color='red')

            ax.text(0.05, 0.95, f"J/S = {j_s:.1f} dB",
                    transform=ax.transAxes, fontsize=8, fontweight='bold', verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            status_text = "Jamming success" if success else "Jamming failure"
            status_color = 'lightgreen' if success else '#ffcccb'

            ax.text(0.05, 0.87, status_text,
                    transform=ax.transAxes, fontsize=8, fontweight='bold', verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.8))

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


if __name__ == '__main__':
    app = JarsGUI()
    app.mainloop()
