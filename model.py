from math import log10, sqrt
import numpy as np

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

# Free space path loss (FSPL) calculation in dB
# FSPL = 20 * log10(distance_km) + 20 * log10(frequency_mhz) + 32.44
def fspl_db(distance_km, frequency_mhz):
    if distance_km <= 0 or frequency_mhz <= 0:
        return float('inf')
    return 20 * log10(distance_km) + 20 * log10(frequency_mhz) + 32.44

# Calculate the received power in dBm at a given position
# using the transmitter's power and frequency, and the distance to the receiver
def received_power_dbm(tx, rx_pos):
    distance_km = tx.position.distance_to(rx_pos) / 1000.0
    return tx.power_dbm - fspl_db(distance_km, tx.frequency_mhz)

# Calculate the J/S ratio in dB at the receiver's position
# J/S ratio = Jamming power received - Signal power received
def j_s_ratio_db(jammer, transmitter, receiver):
    j_recv = received_power_dbm(jammer, receiver.position)
    s_recv = received_power_dbm(transmitter, receiver.position)
    return j_recv - s_recv

def is_communication_successful(signal_dbm, sensitivity_dbm, j_s_db, j_s_threshold_db):
    if signal_dbm < sensitivity_dbm:
        return False  # Signal too weak
    if j_s_db > j_s_threshold_db:
        return False  # Jamming too strong
    return True

def is_jamming_successful(j_s_db, j_s_threshold_db, signal_dbm, sensitivity_dbm):
    if signal_dbm < sensitivity_dbm:
        return False  # Signal was too weak to begin with — not a jammer’s "win"
    return j_s_db > j_s_threshold_db  # Jammer overpowered the usable signal

class MonteCarloModel:
    """
    Manages and executes the Monte Carlo simulation for J/S analysis.
    """
    def __init__(self, tx_params, rx_params, jammer_params_dist, N):
        """
        Initializes the Monte Carlo simulation setup.

        Args:
            tx_params (dict): Deterministic parameters for the Transmitter.
                                e.g., {'power_dbm': 30, 'freq_mhz': 300, 'pos': Position(0,0,0)}
            rx_params (dict): Deterministic parameters for the Receiver.
                                e.g., {'sensitivity_dbm': -90, 'pos': Position(2000,0,0)}
            jammer_params_dist (dict): Uncertain parameters for the Jammer,
                                        represented by scipy.stats distributions.
                                        e.g., {
                                            'power_dbm': stats.norm(loc=40, scale=2),
                                            'pos_x': stats.uniform(loc=900, scale=200),
                                            'pos_y': stats.norm(loc=500, scale=50),
                                            'pos_z': stats.norm(loc=0, scale=20),
                                            'freq_mhz': 300 # Can be deterministic
                                        }
            N (int): The number of simulation runs.
        """
        self.N = N

        # Store deterministic components
        self.transmitter = RadioSource(
            power_dbm=tx_params['power_dbm'],
            frequency_mhz=tx_params['freq_mhz'],
            position=tx_params['pos']
        )
        self.receiver = Receiver(
            sensitivity_dbm=rx_params['sensitivity_dbm'],
            position=rx_params['pos']
        )

        # Generate N random samples for all uncertain variables at once
        self.jam_power_samples = jammer_params_dist['power_dbm'].rvs(size=N)
        self.jam_x_samples = jammer_params_dist['pos_x'].rvs(size=N)
        self.jam_y_samples = jammer_params_dist['pos_y'].rvs(size=N)
        self.jam_z_samples = jammer_params_dist['pos_z'].rvs(size=N)
        
        # Store jammer frequency (assuming it's deterministic for this run)
        self.jammer_freq = jammer_params_dist['freq_mhz']
        
        # You could also make transmitter power uncertain if needed
        # self.tx_power_samples = tx_params['power_dbm_dist'].rvs(size=N)


    def run_simulation(self):
        """
        Runs the full Monte Carlo simulation.

        Returns:
            np.ndarray: An array containing the J/S ratio result from each run.
        """
        j_s_results = []
        for i in range(self.N):
            # Assemble the jammer for this specific run
            jammer_for_run = RadioSource(
                power_dbm=self.jam_power_samples[i],
                frequency_mhz=self.jammer_freq,
                position=Position(
                    self.jam_x_samples[i],
                    self.jam_y_samples[i],
                    self.jam_z_samples[i]
                )
            )

            # Calculate the J/S ratio for this run
            j_s_run = j_s_ratio_db(jammer_for_run, self.transmitter, self.receiver)
            j_s_results.append(j_s_run)

        return np.array(j_s_results)
