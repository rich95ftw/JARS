from model import RadioSource, Receiver, Position, is_jamming_successful, received_power_dbm


class SimulationController:
    def __init__(self):
        pass

    def create_radio_source(self, power, freq, x, y, z):
        return RadioSource(power_dbm=power, frequency_mhz=freq, position=Position(x, y, z))

    def create_receiver(self, sens, x, y, z):
        return Receiver(sensitivity_dbm=sens, position=Position(x, y, z))

    def run_simulation(self, tx: RadioSource, jammer: RadioSource, rx: Receiver, threshold_db):
        tx_to_rx_power_dbm = received_power_dbm(tx, rx.position)
        jam_to_rx_power_dbm = received_power_dbm(jammer, rx.position)
        j_s_db = jam_to_rx_power_dbm - tx_to_rx_power_dbm
        
        success = is_jamming_successful(j_s_db, threshold_db, tx_to_rx_power_dbm, rx.sensitivity_dbm)

        print(f"Tx to Rx Power: {tx_to_rx_power_dbm:.2f} dBm, Jammer to Rx Power: {jam_to_rx_power_dbm:.2f} dBm")
        print(f"J/S Ratio: {j_s_db:.2f}")
        print(f"Sensitivity: {rx.sensitivity_dbm} dBm")
        print(f"Success: {success}")
        print

        return {
            "j_s_db": j_s_db,
            "tx_recv_dbm": tx_to_rx_power_dbm,
            "jam_recv_dbm": jam_to_rx_power_dbm,
            "success": success
        }

    def get_received_powers(self, tx: RadioSource, jammer: RadioSource, rx: Receiver):
        tx_recv = received_power_dbm(tx, rx.position)
        jam_recv = received_power_dbm(jammer, rx.position)
        return tx_recv, jam_recv
