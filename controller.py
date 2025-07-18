from model import (
    RadioSource, Receiver, Position,
    received_power_dbm,
    is_communication_successful
)

from model import MonteCarloModel, Position
from scipy import stats
import numpy as np

class SimulationController:
    def __init__(self):
        pass

    def create_radio_source(self, power, freq, x, y, z):
        return RadioSource(power_dbm=power, frequency_mhz=freq, position=Position(x, y, z))

    def create_receiver(self, sens, x, y, z):
        return Receiver(sensitivity_dbm=sens, position=Position(x, y, z))

    def run_simulation(self, tx: RadioSource, jammer: RadioSource, rx: Receiver, j_s_threshold_db):
        tx_to_rx_power_dbm = received_power_dbm(tx, rx.position)
        jam_to_rx_power_dbm = received_power_dbm(jammer, rx.position)
        j_s_db = jam_to_rx_power_dbm - tx_to_rx_power_dbm

        communication_success = is_communication_successful(
            signal_dbm=tx_to_rx_power_dbm,
            sensitivity_dbm=rx.sensitivity_dbm,
            j_s_db=j_s_db,
            j_s_threshold_db=j_s_threshold_db
        )

        jamming_success = (tx_to_rx_power_dbm >= rx.sensitivity_dbm) and (j_s_db > j_s_threshold_db)

        print(f"Tx → Rx Power: {tx_to_rx_power_dbm:.2f} dBm")
        print(f"Jam → Rx Power: {jam_to_rx_power_dbm:.2f} dBm")
        print(f"J/S Ratio: {j_s_db:.2f} dB")
        print(f"Rx Sensitivity: {rx.sensitivity_dbm:.2f} dBm")
        print(f"Communication Success: {communication_success}")

        return {
            "j_s_db": j_s_db,
            "tx_recv_dbm": tx_to_rx_power_dbm,
            "jam_recv_dbm": jam_to_rx_power_dbm,
            "communication_success": communication_success
        }

    def get_received_powers(self, tx: RadioSource, jammer: RadioSource, rx: Receiver):
        tx_recv = received_power_dbm(tx, rx.position)
        jam_recv = received_power_dbm(jammer, rx.position)
        return tx_recv, jam_recv

    def run_monte_carlo(self, tx_power, tx_freq, tx_pos,
                        rx_sens, rx_pos,
                        jam_power_mean, jam_power_std,
                        jam_freq, jam_pos_x_dist, jam_pos_y_dist, jam_pos_z_dist,
                        N):
        tx_params = {
            'power_dbm': tx_power,
            'freq_mhz': tx_freq,
            'pos': Position(*tx_pos)
        }

        rx_params = {
            'sensitivity_dbm': rx_sens,
            'pos': Position(*rx_pos)
        }

        jammer_params_dist = {
            'power_dbm': stats.norm(loc=jam_power_mean, scale=jam_power_std),
            'pos_x': jam_pos_x_dist,
            'pos_y': jam_pos_y_dist,
            'pos_z': jam_pos_z_dist,
            'freq_mhz': jam_freq
        }

        model = MonteCarloModel(tx_params, rx_params, jammer_params_dist, N)
        js_array = model.run_simulation()

        return {
            'js_array': js_array,
            'mean_js': np.mean(js_array),
            'percentile_90': np.percentile(js_array, 90),
            'percentile_50': np.percentile(js_array, 50)
        }


