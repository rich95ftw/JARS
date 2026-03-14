from math import inf

import numpy as np
from scipy import stats  # Needed for MonteCarloModel tests

from jars.model import (
    MonteCarloModel,
    Position,
    RadioSource,
    Receiver,
    fspl_db,
    is_communication_successful,
    is_jamming_successful,
    j_s_ratio_db,
    received_power_dbm,
)


# Test cases for the Position class
def test_position_creation():
    pos = Position(1, 2, 3)
    assert pos.x == 1.0
    assert pos.y == 2.0
    assert pos.z == 3.0

def test_position_distance_to():
    pos1 = Position(0, 0, 0)
    pos2 = Position(3, 4, 0)
    assert pos1.distance_to(pos2) == 5.0

    pos3 = Position(1, 1, 1)
    pos4 = Position(1, 1, 1)
    assert pos3.distance_to(pos4) == 0.0

    pos5 = Position(0, 0, 0)
    pos6 = Position(0, 0, 10)
    assert pos5.distance_to(pos6) == 10.0

# Test cases for RadioSource class
def test_radio_source_creation():
    pos = Position(0, 0, 0)
    source = RadioSource(power_dbm=30, frequency_mhz=100, position=pos)
    assert source.power_dbm == 30.0
    assert source.frequency_mhz == 100.0
    assert source.position == pos

# Test cases for Receiver class
def test_receiver_creation():
    pos = Position(10, 20, 30)
    receiver = Receiver(sensitivity_dbm=-80, position=pos)
    assert receiver.sensitivity_dbm == -80.0
    assert receiver.position == pos

# Test cases for fspl_db function
def test_fspl_db_valid_inputs():
    # Example values, verify with a calculator if needed
    assert abs(fspl_db(1, 100) - (20 * np.log10(1) + 20 * np.log10(100) + 32.44)) < 1e-9
    assert abs(fspl_db(10, 500) - (20 * np.log10(10) + 20 * np.log10(500) + 32.44)) < 1e-9

def test_fspl_db_zero_or_negative_distance():
    assert fspl_db(0, 100) == float('inf')
    assert fspl_db(-1, 100) == float('inf')

def test_fspl_db_zero_or_negative_frequency():
    assert fspl_db(1, 0) == float('inf')
    assert fspl_db(1, -100) == float('inf')

# Test cases for received_power_dbm function
def test_received_power_dbm():
    tx_pos = Position(0, 0, 0)
    tx = RadioSource(power_dbm=30, frequency_mhz=100, position=tx_pos)
    rx_pos = Position(1000, 0, 0) # 1 km distance

    # Manually calculate expected FSPL for 1km, 100MHz
    expected_fspl = 20 * np.log10(1) + 20 * np.log10(100) + 32.44
    expected_received_power = 30 - expected_fspl

    assert abs(received_power_dbm(tx, rx_pos) - expected_received_power) < 1e-9

def test_received_power_dbm_tx_antenna_gain():
    """Tx antenna gain adds directly to received power."""
    tx_pos = Position(0, 0, 0)
    rx_pos = Position(1000, 0, 0)
    tx_no_gain = RadioSource(power_dbm=30, frequency_mhz=100, position=tx_pos)
    tx_with_gain = RadioSource(power_dbm=30, frequency_mhz=100,
                               position=tx_pos, antenna_gain_dbi=10.0)
    diff = received_power_dbm(tx_with_gain, rx_pos) - received_power_dbm(tx_no_gain, rx_pos)
    assert abs(diff - 10.0) < 1e-9


def test_received_power_dbm_rx_antenna_gain():
    """Rx antenna gain adds directly to received power."""
    tx_pos = Position(0, 0, 0)
    rx_pos = Position(1000, 0, 0)
    tx = RadioSource(power_dbm=30, frequency_mhz=100, position=tx_pos)
    diff = (received_power_dbm(tx, rx_pos, rx_antenna_gain_dbi=6.0)
            - received_power_dbm(tx, rx_pos))
    assert abs(diff - 6.0) < 1e-9


def test_j_s_ratio_rx_gain_cancels():
    """Rx antenna gain affects both paths equally so J/S is unchanged."""
    jammer = RadioSource(power_dbm=50, frequency_mhz=300,
                         position=Position(500, 0, 0))
    tx = RadioSource(power_dbm=30, frequency_mhz=300,
                     position=Position(0, 0, 0))
    rx_no_gain = Receiver(sensitivity_dbm=-90, position=Position(1000, 0, 0))
    rx_with_gain = Receiver(sensitivity_dbm=-90, position=Position(1000, 0, 0),
                            antenna_gain_dbi=12.0)
    assert abs(j_s_ratio_db(jammer, tx, rx_no_gain)
               - j_s_ratio_db(jammer, tx, rx_with_gain)) < 1e-9


def test_received_power_dbm_same_position():
    tx_pos = Position(0, 0, 0)
    tx = RadioSource(power_dbm=30, frequency_mhz=100, position=tx_pos)
    rx_pos = Position(0, 0, 0) # Same position, distance is 0, should be inf FSPL

    # When distance is 0, fspl_db returns inf, leading to -inf received power
    assert received_power_dbm(tx, rx_pos) == -inf


# Test cases for j_s_ratio_db function
def test_j_s_ratio_db():
    # Setup a scenario where J/S can be calculated
    jammer_pos = Position(500, 0, 0)
    jammer = RadioSource(power_dbm=50, frequency_mhz=300, position=jammer_pos)

    tx_pos = Position(0, 0, 0)
    transmitter = RadioSource(power_dbm=30, frequency_mhz=300, position=tx_pos)

    receiver_pos = Position(1000, 0, 0)
    receiver = Receiver(sensitivity_dbm=-90, position=receiver_pos)

    # Calculate expected received power for jammer
    jammer_dist_km = jammer_pos.distance_to(receiver_pos) / 1000.0 # 0.5 km
    jammer_fspl = fspl_db(jammer_dist_km, jammer.frequency_mhz)
    j_recv_expected = jammer.power_dbm - jammer_fspl

    # Calculate expected received power for signal
    signal_dist_km = tx_pos.distance_to(receiver_pos) / 1000.0 # 1 km
    signal_fspl = fspl_db(signal_dist_km, transmitter.frequency_mhz)
    s_recv_expected = transmitter.power_dbm - signal_fspl

    expected_j_s_ratio = j_recv_expected - s_recv_expected

    assert abs(j_s_ratio_db(jammer, transmitter, receiver) - expected_j_s_ratio) < 1e-9

# Test cases for is_communication_successful
def test_is_communication_successful():
    # Signal strong enough, jamming below threshold
    assert is_communication_successful(signal_dbm=-70, sensitivity_dbm=-80, j_s_db=5, j_s_threshold_db=10) == True
    # Signal too weak
    assert is_communication_successful(signal_dbm=-90, sensitivity_dbm=-80, j_s_db=5, j_s_threshold_db=10) == False
    # Jamming too strong
    assert is_communication_successful(signal_dbm=-70, sensitivity_dbm=-80, j_s_db=15, j_s_threshold_db=10) == False
    # Signal too weak AND jamming too strong
    assert is_communication_successful(signal_dbm=-90, sensitivity_dbm=-80, j_s_db=15, j_s_threshold_db=10) == False
    # Borderline cases
    assert is_communication_successful(signal_dbm=-80, sensitivity_dbm=-80, j_s_db=10, j_s_threshold_db=10) == True
    assert is_communication_successful(signal_dbm=-81, sensitivity_dbm=-80, j_s_db=10, j_s_threshold_db=10) == False # Signal just too weak
    assert is_communication_successful(signal_dbm=-80, sensitivity_dbm=-80, j_s_db=10.0000000001, j_s_threshold_db=10) == False # Jamming just too strong

# Test cases for is_jamming_successful
def test_is_jamming_successful():
    # Jamming successful
    assert is_jamming_successful(j_s_db=15, j_s_threshold_db=10, signal_dbm=-70, sensitivity_dbm=-80) == True
    # Jamming not successful (below threshold)
    assert is_jamming_successful(j_s_db=5, j_s_threshold_db=10, signal_dbm=-70, sensitivity_dbm=-80) == False
    # Jamming not successful (signal was too weak to begin with)
    assert is_jamming_successful(j_s_db=15, j_s_threshold_db=10, signal_dbm=-90, sensitivity_dbm=-80) == False
    # Borderline cases
    assert is_jamming_successful(j_s_db=10.0000000001, j_s_threshold_db=10, signal_dbm=-70, sensitivity_dbm=-80) == True
    assert is_jamming_successful(j_s_db=10, j_s_threshold_db=10, signal_dbm=-70, sensitivity_dbm=-80) == False # Not strictly greater

# Test cases for MonteCarloModel
def _make_rx_params(N, x=2000, y=0, z=0, x_std=0, y_std=0, z_std=0):
    def _samp(loc, scale):
        if scale == 0:
            return np.full(N, float(loc))
        return stats.norm(loc=loc, scale=scale).rvs(size=N)
    return {
        'sensitivity_dbm': -90,
        'pos_x': _samp(x, x_std),
        'pos_y': _samp(y, y_std),
        'pos_z': _samp(z, z_std),
    }

def test_monte_carlo_model_initialization():
    tx_params = {'power_dbm': 30, 'freq_mhz': 300, 'pos': Position(0,0,0)}
    jammer_params_dist = {
        'power_dbm': stats.norm(loc=40, scale=2),
        'pos_x': stats.uniform(loc=900, scale=200),
        'pos_y': stats.norm(loc=500, scale=50),
        'pos_z': stats.norm(loc=0, scale=20),
        'freq_mhz': 300
    }
    N = 100
    rx_params = _make_rx_params(N)

    model = MonteCarloModel(tx_params, rx_params, jammer_params_dist, N)

    assert model.N == N
    assert isinstance(model.transmitter, RadioSource)
    assert model.rx_sensitivity == -90.0
    assert len(model.rx_x_samples) == N
    assert len(model.rx_y_samples) == N
    assert len(model.rx_z_samples) == N
    assert len(model.jam_power_samples) == N
    assert len(model.jam_x_samples) == N
    assert len(model.jam_y_samples) == N
    assert len(model.jam_z_samples) == N
    assert model.jammer_freq == 300.0

def test_monte_carlo_model_run_simulation():
    tx_params = {'power_dbm': 30, 'freq_mhz': 300, 'pos': Position(0,0,0)}
    jammer_params_dist = {
        'power_dbm': stats.norm(loc=40, scale=2),
        'pos_x': stats.uniform(loc=900, scale=200),
        'pos_y': stats.norm(loc=500, scale=50),
        'pos_z': stats.norm(loc=0, scale=20),
        'freq_mhz': 300
    }
    N = 100
    rx_params = _make_rx_params(N)

    model = MonteCarloModel(tx_params, rx_params, jammer_params_dist, N)
    js_array, tx_recv_array = model.run_simulation()

    assert isinstance(js_array, np.ndarray)
    assert js_array.shape == (N,)
    assert np.all(np.isfinite(js_array))
    assert isinstance(tx_recv_array, np.ndarray)
    assert tx_recv_array.shape == (N,)
    assert np.all(np.isfinite(tx_recv_array))

def test_monte_carlo_model_run_simulation_with_rx_uncertainty():
    tx_params = {'power_dbm': 30, 'freq_mhz': 300, 'pos': Position(0,0,0)}
    jammer_params_dist = {
        'power_dbm': stats.norm(loc=40, scale=2),
        'pos_x': stats.uniform(loc=900, scale=200),
        'pos_y': stats.norm(loc=500, scale=50),
        'pos_z': stats.norm(loc=0, scale=20),
        'freq_mhz': 300
    }
    N = 200
    rx_params = _make_rx_params(N, x=2000, y=0, z=0, x_std=100, y_std=50)

    model = MonteCarloModel(tx_params, rx_params, jammer_params_dist, N)
    js_array, tx_recv_array = model.run_simulation()

    assert js_array.shape == (N,)
    assert tx_recv_array.shape == (N,)
    # With Rx position uncertainty tx_recv_array should vary across runs
    assert tx_recv_array.std() > 0
    assert np.all(np.isfinite(js_array))
    assert np.all(np.isfinite(tx_recv_array))

    # Test with N=1
    rx_params_single = _make_rx_params(1)
    model_single = MonteCarloModel(tx_params, rx_params_single, jammer_params_dist, 1)
    js_single, tx_single = model_single.run_simulation()
    assert js_single.shape == (1,)
    assert tx_single.shape == (1,)
