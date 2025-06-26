from math import log10, sqrt


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

# Check if jamming is successful based on the J/S ratio, threshold, and receiver sensitivity
# Returns True if jamming is successful, False otherwise
def is_jamming_successful(j_s_db, threshold_db, signal_dbm, sensitivity_dbm):
    if signal_dbm < sensitivity_dbm:
        return False  # Receiver can't detect the signal at all
    return j_s_db > threshold_db



