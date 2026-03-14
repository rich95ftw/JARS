import numpy as np
from scipy import stats

from jars.model import (
    MonteCarloModel,
    Position,
    RadioSource,
    Receiver,
    is_communication_successful,
    is_jamming_successful,
    j_s_ratio_db,
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
                            y: float, z: float,
                            antenna_gain_dbi: float = 0.0) -> RadioSource:
        """
        Creates a RadioSource object.

        Args:
            power (float): Transmit power in dBm.
            freq (float): Frequency in MHz.
            x (float): X-coordinate of the source's position.
            y (float): Y-coordinate of the source's position.
            z (float): Z-coordinate of the source's position.
            antenna_gain_dbi (float): Antenna gain in dBi. Default 0.

        Returns:
            RadioSource: A new RadioSource instance.
        """
        return RadioSource(power_dbm=power, frequency_mhz=freq,
                           position=Position(x, y, z),
                           antenna_gain_dbi=antenna_gain_dbi)

    def create_receiver(self, sens: float, x: float, y: float,
                        z: float, antenna_gain_dbi: float = 0.0) -> Receiver:
        """
        Creates a Receiver object.

        Args:
            sens (float): Receiver sensitivity in dBm.
            x (float): X-coordinate of the receiver's position.
            y (float): Y-coordinate of the receiver's position.
            z (float): Z-coordinate of the receiver's position.
            antenna_gain_dbi (float): Antenna gain in dBi. Default 0.

        Returns:
            Receiver: A new Receiver instance.
        """
        return Receiver(sensitivity_dbm=sens, position=Position(x, y, z),
                        antenna_gain_dbi=antenna_gain_dbi)

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
        frequency_mismatch: bool = jammer.frequency_mhz != tx.frequency_mhz

        tx_to_rx_power_dbm: float = received_power_dbm(tx, rx.position,
                                                        rx.antenna_gain_dbi)
        jam_to_rx_power_dbm: float = received_power_dbm(jammer, rx.position,
                                                         rx.antenna_gain_dbi)
        j_s_db: float = j_s_ratio_db(jammer, tx, rx)

        communication_success: bool = is_communication_successful(
            signal_dbm=tx_to_rx_power_dbm,
            sensitivity_dbm=rx.sensitivity_dbm,
            j_s_db=j_s_db,
            j_s_threshold_db=j_s_threshold_db
        )

        jamming_success: bool = is_jamming_successful(
            j_s_db=j_s_db,
            j_s_threshold_db=j_s_threshold_db,
            signal_dbm=tx_to_rx_power_dbm,
            sensitivity_dbm=rx.sensitivity_dbm,
        )

        print(f"Tx → Rx Power: {tx_to_rx_power_dbm:.2f} dBm")
        print(f"Jam → Rx Power: {jam_to_rx_power_dbm:.2f} dBm")
        print(f"J/S Ratio: {j_s_db:.2f} dB")
        print(f"Rx Sensitivity: {rx.sensitivity_dbm:.2f} dBm")
        print(f"Communication Success: {communication_success}")
        print(f"Jamming Success: {jamming_success}")
        if frequency_mismatch:
            print(
                f"WARNING: Jammer frequency ({jammer.frequency_mhz} MHz) does "
                f"not match TX frequency ({tx.frequency_mhz} MHz). "
                f"J/S result may not be physically valid for spot jamming."
            )

        return {
            "j_s_db": j_s_db,
            "tx_recv_dbm": tx_to_rx_power_dbm,
            "jam_recv_dbm": jam_to_rx_power_dbm,
            "communication_success": communication_success,
            "jamming_success": jamming_success,
            "frequency_mismatch": frequency_mismatch,
        }

    def run_monte_carlo(self, tx_power: float, tx_freq: float,
                        tx_pos: tuple[float, float, float],
                        rx_sens: float, rx_pos: tuple[float, float, float],
                        jam_power_mean: float, jam_power_std: float,
                        jam_freq: float, jam_pos_x_dist: stats.rv_continuous,
                        jam_pos_y_dist: stats.rv_continuous,
                        jam_pos_z_dist: stats.rv_continuous,
                        j_s_threshold_db: float,
                        N: int,
                        rx_x_std: float = 0.0,
                        rx_y_std: float = 0.0,
                        rx_z_std: float = 0.0,
                        jam_power_min: float = -np.inf,
                        jam_power_max: float = np.inf,
                        tx_antenna_gain_dbi: float = 0.0,
                        jammer_antenna_gain_dbi: float = 0.0,
                        rx_antenna_gain_dbi: float = 0.0) -> dict:
        """
        Runs a Monte Carlo simulation to analyze the J/S ratio distribution.

        Args:
            tx_power (float): Mean transmit power of the transmitter in dBm.
            tx_freq (float): Frequency of the transmitter in MHz.
            tx_pos (tuple[float, float, float]): (x, y, z) coordinates of the
                                                 transmitter.
            rx_sens (float): Receiver sensitivity in dBm.
            rx_pos (tuple[float, float, float]): Nominal (x, y, z) coordinates
                                                 of the receiver.
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
            j_s_threshold_db (float): The J/S ratio threshold in dB for
                                      successful communication.
            N (int): Number of Monte Carlo iterations.
            rx_x_std (float): Std deviation of receiver X position (m).
                               Default 0 (deterministic).
            rx_y_std (float): Std deviation of receiver Y position (m).
                               Default 0 (deterministic).
            rx_z_std (float): Std deviation of receiver Z position (m).
                               Default 0 (deterministic).
            jam_power_min (float): Lower bound for jammer power (dBm).
                                   Default -inf (no lower truncation).
            jam_power_max (float): Upper bound for jammer power (dBm).
                                   Default +inf (no upper truncation).

        Returns:
            dict: A dictionary containing Monte Carlo simulation results:
                - "js_array" (np.ndarray): Array of J/S ratio values from each
                                           iteration.
                - "mean_js" (float): Mean J/S ratio.
                - "percentile_90" (float): 90th percentile of J/S ratio.
                - "percentile_50" (float): 50th percentile (median) of J/S
                                           ratio.
                - "tx_recv_dbm" (float): Mean received signal power at the
                                         receiver across all runs.
                - "p_jamming_success" (float): Fraction of runs in which
                                               jamming succeeded.
                - "p_comm_success" (float): Fraction of runs in which
                                            communication succeeded.
                - "rx_position_uncertain" (bool): True if any Rx position
                                                  std > 0.
        """
        def _pos_samples(loc: float, scale: float) -> np.ndarray:
            if scale == 0.0:
                return np.full(N, loc)
            return stats.norm(loc=loc, scale=scale).rvs(size=N)

        tx_params: dict = {
            'power_dbm': tx_power,
            'freq_mhz': tx_freq,
            'pos': Position(*tx_pos),
            'antenna_gain_dbi': tx_antenna_gain_dbi,
        }

        rx_params: dict = {
            'sensitivity_dbm': rx_sens,
            'antenna_gain_dbi': rx_antenna_gain_dbi,
            'pos_x': _pos_samples(rx_pos[0], rx_x_std),
            'pos_y': _pos_samples(rx_pos[1], rx_y_std),
            'pos_z': _pos_samples(rx_pos[2], rx_z_std),
        }

        if jam_power_std <= 0.0:
            power_dist = stats.norm(loc=jam_power_mean, scale=1e-6)
        else:
            a = (jam_power_min - jam_power_mean) / jam_power_std
            b = (jam_power_max - jam_power_mean) / jam_power_std
            power_dist = stats.truncnorm(
                a=a, b=b, loc=jam_power_mean, scale=jam_power_std
            )

        jammer_params_dist: dict = {
            'power_dbm': power_dist,
            'pos_x': jam_pos_x_dist,
            'pos_y': jam_pos_y_dist,
            'pos_z': jam_pos_z_dist,
            'freq_mhz': jam_freq,
            'antenna_gain_dbi': jammer_antenna_gain_dbi,
        }

        model = MonteCarloModel(tx_params, rx_params, jammer_params_dist, N)
        js_array, tx_recv_array = model.run_simulation()

        frequency_mismatch: bool = jam_freq != tx_freq
        rx_position_uncertain: bool = any(s > 0.0 for s in
                                          (rx_x_std, rx_y_std, rx_z_std))

        # Evaluate success per run: signal must be receivable AND J/S in range.
        signal_receivable_array = tx_recv_array >= rx_sens
        comm_success_array = signal_receivable_array & (js_array <= j_s_threshold_db)
        jam_success_array = signal_receivable_array & (js_array > j_s_threshold_db)

        return {
            'js_array': js_array,
            'mean_js': float(np.mean(js_array)),
            'percentile_90': float(np.percentile(js_array, 90)),
            'percentile_50': float(np.percentile(js_array, 50)),
            'tx_recv_dbm': float(np.mean(tx_recv_array)),
            'p_jamming_success': float(np.mean(jam_success_array)),
            'p_comm_success': float(np.mean(comm_success_array)),
            'frequency_mismatch': frequency_mismatch,
            'rx_position_uncertain': rx_position_uncertain,
        }
