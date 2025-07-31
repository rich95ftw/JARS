import numpy as np
from scipy import stats

from model import (
    MonteCarloModel,
    Position,
    RadioSource,
    Receiver,
    is_communication_successful,
    received_power_dbm,
)


class SimulationController:
    """
    Controls the simulation of radio communication and jamming scenarios.
    """

    def __init__(self) -> None:
        """
        Initializes the SimulationController.
        """
        pass

    def create_radio_source(self, power: float, freq: float, x: float,
                            y: float, z: float) -> RadioSource:
        """
        Creates a RadioSource object.

        Args:
            power (float): Transmit power in dBm.
            freq (float): Frequency in MHz.
            x (float): X-coordinate of the source's position.
            y (float): Y-coordinate of the source's position.
            z (float): Z-coordinate of the source's position.

        Returns:
            RadioSource: A new RadioSource instance.
        """
        return RadioSource(power_dbm=power, frequency_mhz=freq,
                           position=Position(x, y, z))

    def create_receiver(self, sens: float, x: float, y: float,
                        z: float) -> Receiver:
        """
        Creates a Receiver object.

        Args:
            sens (float): Receiver sensitivity in dBm.
            x (float): X-coordinate of the receiver's position.
            y (float): Y-coordinate of the receiver's position.
            z (float): Z-coordinate of the receiver's position.

        Returns:
            Receiver: A new Receiver instance.
        """
        return Receiver(sensitivity_dbm=sens, position=Position(x, y, z))

    def run_simulation(self, tx: RadioSource, jammer: RadioSource,
                       rx: Receiver, j_s_threshold_db: float) -> dict:
        """
        Runs a single simulation of communication with a jammer.

        Args:
            tx (RadioSource): The transmitting radio source.
            jammer (RadioSource): The jamming radio source.
            rx (Receiver): The receiving radio.
            j_s_threshold_db (float): The J/S ratio threshold in dB
                                      for successful communication.

        Returns:
            dict: A dictionary containing simulation results:
                - "j_s_db" (float): Jammer-to-Signal ratio in dB.
                - "tx_recv_dbm" (float): Received power from the transmitter
                                         in dBm.
                - "jam_recv_dbm" (float): Received power from the jammer
                                          in dBm.
                - "communication_success" (bool): True if communication is
                                                  successful, False otherwise.
        """
        tx_to_rx_power_dbm: float = received_power_dbm(tx, rx.position)
        jam_to_rx_power_dbm: float = received_power_dbm(jammer, rx.position)
        j_s_db: float = jam_to_rx_power_dbm - tx_to_rx_power_dbm

        communication_success: bool = is_communication_successful(
            signal_dbm=tx_to_rx_power_dbm,
            sensitivity_dbm=rx.sensitivity_dbm,
            j_s_db=j_s_db,
            j_s_threshold_db=j_s_threshold_db
        )

        # Jamming success implies communication is possible
        # (tx_to_rx_power_dbm >= rx.sensitivity_dbm) AND the jamming is
        # effective (j_s_db > j_s_threshold_db).
        jamming_success: bool = (tx_to_rx_power_dbm >= rx.sensitivity_dbm) and \
                                (j_s_db > j_s_threshold_db)

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

    def get_received_powers(self, tx: RadioSource, jammer: RadioSource,
                            rx: Receiver) -> tuple[float, float]:
        """
        Calculates the received power from the transmitter and the jammer at
        the receiver.

        Args:
            tx (RadioSource): The transmitting radio source.
            jammer (RadioSource): The jamming radio source.
            rx (Receiver): The receiving radio.

        Returns:
            tuple[float, float]: A tuple containing:
                - The received power from the transmitter in dBm.
                - The received power from the jammer in dBm.
        """
        tx_recv: float = received_power_dbm(tx, rx.position)
        jam_recv: float = received_power_dbm(jammer, rx.position)
        return tx_recv, jam_recv

    def run_monte_carlo(self, tx_power: float, tx_freq: float,
                        tx_pos: tuple[float, float, float],
                        rx_sens: float, rx_pos: tuple[float, float, float],
                        jam_power_mean: float, jam_power_std: float,
                        jam_freq: float, jam_pos_x_dist: stats.rv_continuous,
                        jam_pos_y_dist: stats.rv_continuous,
                        jam_pos_z_dist: stats.rv_continuous,
                        N: int) -> dict:
        """
        Runs a Monte Carlo simulation to analyze the J/S ratio distribution.

        Args:
            tx_power (float): Mean transmit power of the transmitter in dBm.
            tx_freq (float): Frequency of the transmitter in MHz.
            tx_pos (tuple[float, float, float]): (x, y, z) coordinates of the
                                                 transmitter.
            rx_sens (float): Receiver sensitivity in dBm.
            rx_pos (tuple[float, float, float]): (x, y, z) coordinates of the
                                                 receiver.
            jam_power_mean (float): Mean power of the jammer in dBm.
            jam_power_std (float): Standard deviation of the jammer's power
                                   in dBm.
            jam_freq (float): Frequency of the jammer in MHz.
            jam_pos_x_dist (stats.rv_continuous): SciPy statistical
                                                  distribution for jammer's
                                                  X position.
            jam_pos_y_dist (stats.rv_continuous): SciPy statistical
                                                  distribution for jammer's
                                                  Y position.
            jam_pos_z_dist (stats.rv_continuous): SciPy statistical
                                                  distribution for jammer's
                                                  Z position.
            N (int): Number of Monte Carlo iterations.

        Returns:
            dict: A dictionary containing Monte Carlo simulation results:
                - "js_array" (np.ndarray): Array of J/S ratio values from each
                                           iteration.
                - "mean_js" (float): Mean J/S ratio.
                - "percentile_90" (float): 90th percentile of J/S ratio.
                - "percentile_50" (float): 50th percentile (median) of J/S
                                           ratio.
        """
        tx_params: dict = {
            'power_dbm': tx_power,
            'freq_mhz': tx_freq,
            'pos': Position(*tx_pos)
        }

        rx_params: dict = {
            'sensitivity_dbm': rx_sens,
            'pos': Position(*rx_pos)
        }

        jammer_params_dist: dict = {
            'power_dbm': stats.norm(loc=jam_power_mean, scale=jam_power_std),
            'pos_x': jam_pos_x_dist,
            'pos_y': jam_pos_y_dist,
            'pos_z': jam_pos_z_dist,
            'freq_mhz': jam_freq
        }

        model = MonteCarloModel(tx_params, rx_params, jammer_params_dist, N)
        js_array: np.ndarray = model.run_simulation()

        return {
            'js_array': js_array,
            'mean_js': np.mean(js_array),
            'percentile_90': np.percentile(js_array, 90),
            'percentile_50': np.percentile(js_array, 50)
        }