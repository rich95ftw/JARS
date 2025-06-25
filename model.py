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


def fspl_db(distance_km, frequency_mhz):
    if distance_km <= 0 or frequency_mhz <= 0:
        return float('inf')
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
