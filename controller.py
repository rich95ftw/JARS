from model import RadioSource, Receiver, Position, j_s_ratio_db, is_jamming_successful, received_power_dbm


class SimulationController:
    def __init__(self):
        pass

    def create_radio_source(self, power, freq, x, y, z):
        return RadioSource(power_dbm=power, frequency_mhz=freq, position=Position(x, y, z))

    def create_receiver(self, sens, x, y, z):
        return Receiver(sensitivity_dbm=sens, position=Position(x, y, z))

    def run_simulation(self, tx: RadioSource, jammer: RadioSource, rx: Receiver, threshold_db):
        j_s = j_s_ratio_db(jammer, tx, rx)
        success = is_jamming_successful(j_s, threshold_db)
        return j_s, success

    def get_received_powers(self, tx: RadioSource, jammer: RadioSource, rx: Receiver):
        tx_recv = received_power_dbm(tx, rx.position)
        jam_recv = received_power_dbm(jammer, rx.position)
        return tx_recv, jam_recv
