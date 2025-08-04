from __future__ import annotations

from math import log10, sqrt
from typing import Any

import numpy as np


class Position:
    """Represents a 3D position in space."""

    def __init__(self, x: float | int, y: float | int,
                 z: float | int) -> None:
        """
        Initializes a Position object.

        Args:
            x (float | int): The x-coordinate.
            y (float | int): The y-coordinate.
            z (float | int): The z-coordinate.
        """
        self.x: float = float(x)
        self.y: float = float(y)
        self.z: float = float(z)

    def distance_to(self, other: Position) -> float:
        """
        Calculates the Euclidean distance to another Position object.

        Args:
            other (Position): The other Position object.

        Returns:
            float: The distance between the two positions.
        """
        return sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )


class RadioSource:
    """Represents a radio source with transmit power, frequency, and position."""

    def __init__(self, power_dbm: float | int,
                 frequency_mhz: float | int,
                 position: Position) -> None:
        """
        Initializes a RadioSource.

        Args:
            power_dbm (Union[float, int]): The transmit power in dBm.
            frequency_mhz (Union[float, int]): The frequency in MHz.
            position (Position): The position of the radio source.
        """
        self.power_dbm: float = float(power_dbm)
        self.frequency_mhz: float = float(frequency_mhz)
        self.position: Position = position


class Receiver:
    """Represents a radio receiver with a specific sensitivity and position."""

    def __init__(self, sensitivity_dbm: float | int,
                 position: Position) -> None:
        """
        Initializes a Receiver.

        Args:
            sensitivity_dbm (Union[float, int]): The receiver sensitivity in dBm.
            position (Position): The position of the receiver.
        """
        self.sensitivity_dbm: float = float(sensitivity_dbm)
        self.position: Position = position


def fspl_db(distance_km: float, frequency_mhz: float) -> float:
    """
    Calculates Free Space Path Loss (FSPL) in dB.

    FSPL = 20 * log10(distance_km) + 20 * log10(frequency_mhz) + 32.44

    Args:
        distance_km (float): The distance in kilometers.
        frequency_mhz (float): The frequency in megahertz.

    Returns:
        float: The path loss in dB. Returns infinity if distance or
               frequency is not positive.
    """
    if distance_km <= 0 or frequency_mhz <= 0:
        return float("inf")
    return (20 * log10(distance_km) + 20 * log10(frequency_mhz) + 32.44)


def received_power_dbm(tx: RadioSource, rx_pos: Position) -> float:
    """
    Calculates the received power in dBm at a given position.

    Args:
        tx (RadioSource): The transmitting radio source.
        rx_pos (Position): The position of the receiver.

    Returns:
        float: The received power in dBm.
    """
    distance_km: float = tx.position.distance_to(rx_pos) / 1000.0
    return tx.power_dbm - fspl_db(distance_km, tx.frequency_mhz)


def j_s_ratio_db(jammer: RadioSource, transmitter: RadioSource,
                 receiver: Receiver) -> float:
    """
    Calculates the Jammer-to-Signal (J/S) ratio in dB at the receiver's position.

    Args:
        jammer (RadioSource): The jamming radio source.
        transmitter (RadioSource): The signal transmitting radio source.
        receiver (Receiver): The receiving radio.

    Returns:
        float: The J/S ratio in dB.
    """
    j_recv: float = received_power_dbm(jammer, receiver.position)
    s_recv: float = received_power_dbm(transmitter, receiver.position)
    return j_recv - s_recv


def is_communication_successful(signal_dbm: float, sensitivity_dbm: float,
                                j_s_db: float, j_s_threshold_db: float) -> bool:
    """
    Determines if communication is successful based on signal strength and jamming.

    Args:
        signal_dbm (float): The received signal power in dBm.
        sensitivity_dbm (float): The receiver's sensitivity in dBm.
        j_s_db (float): The Jammer-to-Signal ratio in dB.
        j_s_threshold_db (float): The J/S ratio threshold for successful
                                  communication.

    Returns:
        bool: True if communication is successful, False otherwise.
    """
    if signal_dbm < sensitivity_dbm:
        return False  # Signal is too weak to be received
    if j_s_db > j_s_threshold_db:
        return False  # Jamming is too strong
    return True


def is_jamming_successful(j_s_db: float, j_s_threshold_db: float,
                          signal_dbm: float, sensitivity_dbm: float) -> bool:
    """
    Determines if jamming is successful (i.e., it prevented communication).

    Args:
        j_s_db (float): The Jammer-to-Signal ratio in dB.
        j_s_threshold_db (float): The J/S ratio threshold for successful
                                  communication.
        signal_dbm (float): The received signal power in dBm.
        sensitivity_dbm (float): The receiver's sensitivity in dBm.

    Returns:
        bool: True if jamming was successful, False otherwise.
    """
    if signal_dbm < sensitivity_dbm:
        return False  # Jammer can't disrupt a signal that couldn't be received
    return j_s_db > j_s_threshold_db


class MonteCarloModel:
    """
    Manages and executes the Monte Carlo simulation for J/S analysis.
    """

    def __init__(self, tx_params: dict[str, Any], rx_params: dict[str, Any],
                 jammer_params_dist: dict[str, Any], N: int) -> None:
        """
        Initializes the Monte Carlo simulation setup.

        Args:
            tx_params (Dict[str, Any]): Deterministic parameters for the
                                         Transmitter.
                                         e.g., {'power_dbm': 30,
                                                'freq_mhz': 300,
                                                'pos': Position(0,0,0)}
            rx_params (Dict[str, Any]): Deterministic parameters for the
                                         Receiver.
                                         e.g., {'sensitivity_dbm': -90,
                                                'pos': Position(2000,0,0)}
            jammer_params_dist (Dict[str, Any]): Uncertain parameters for the
                                                 Jammer, represented by
                                                 scipy.stats distributions.
                                        e.g., {
                                            'power_dbm': stats.norm(loc=40,
                                                                    scale=2),
                                            'pos_x': stats.uniform(loc=900,
                                                                   scale=200),
                                            'pos_y': stats.norm(loc=500,
                                                                scale=50),
                                            'pos_z': stats.norm(loc=0, scale=20),
                                            'freq_mhz': 300 # Can be
                                                          # deterministic
                                        }
            N (int): The number of simulation runs.
        """
        self.N: int = N
        self.transmitter: RadioSource = RadioSource(
            power_dbm=tx_params["power_dbm"],
            frequency_mhz=tx_params["freq_mhz"],
            position=tx_params["pos"],
        )
        self.receiver: Receiver = Receiver(
            sensitivity_dbm=rx_params["sensitivity_dbm"],
            position=rx_params["pos"]
        )
        self.jam_power_samples: np.ndarray = (
            jammer_params_dist["power_dbm"].rvs(size=N)
        )
        self.jam_x_samples: np.ndarray = (
            jammer_params_dist["pos_x"].rvs(size=N)
        )
        self.jam_y_samples: np.ndarray = (
            jammer_params_dist["pos_y"].rvs(size=N)
        )
        self.jam_z_samples: np.ndarray = (
            jammer_params_dist["pos_z"].rvs(size=N)
        )
        self.jammer_freq: float = jammer_params_dist["freq_mhz"]

    def run_simulation(self) -> np.ndarray:
        """
        Runs the full Monte Carlo simulation.

        Returns:
            np.ndarray: An array containing the J/S ratio result from each run.
        """
        j_s_results: list[float] = []
        for i in range(self.N):
            jammer_for_run: RadioSource = RadioSource(
                power_dbm=self.jam_power_samples[i],
                frequency_mhz=self.jammer_freq,
                position=Position(
                    self.jam_x_samples[i],
                    self.jam_y_samples[i],
                    self.jam_z_samples[i]
                ),
            )
            j_s_run: float = j_s_ratio_db(
                jammer_for_run,
                self.transmitter,
                self.receiver
            )
            j_s_results.append(j_s_run)
        return np.array(j_s_results)