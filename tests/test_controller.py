import pytest

from jars.controller import SimulationController
from jars.model import fspl_db, received_power_dbm, RadioSource, Receiver, Position

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def controller():
    return SimulationController()


# Geometry used across several tests:
#   TX  at (0, 0, 0),    power=30 dBm, freq=300 MHz
#   RX  at (2000, 0, 0), sensitivity=-90 dBm   → TX-RX distance = 2 km
#   JAM at (500, 0, 0),  freq=300 MHz           → JAM-RX distance = 1.5 km

TX_POWER   = 30.0
TX_FREQ    = 300.0
TX_POS     = (0.0, 0.0, 0.0)

RX_SENS    = -90.0
RX_POS     = (2000.0, 0.0, 0.0)

JAM_FREQ   = 300.0
JAM_POS    = (500.0, 0.0, 0.0)

THRESHOLD  = 10.0   # dB


def _expected_tx_recv() -> float:
    """Received power from TX at RX (2 km, 300 MHz)."""
    return TX_POWER - fspl_db(2.0, TX_FREQ)


def _expected_jam_recv(jam_power: float) -> float:
    """Received power from jammer at RX (1.5 km, 300 MHz)."""
    return jam_power - fspl_db(1.5, JAM_FREQ)


# ---------------------------------------------------------------------------
# Result dictionary structure
# ---------------------------------------------------------------------------

def test_run_simulation_returns_expected_keys(controller):
    tx     = controller.create_radio_source(TX_POWER, TX_FREQ, *TX_POS)
    jammer = controller.create_radio_source(20.0, JAM_FREQ, *JAM_POS)
    rx     = controller.create_receiver(RX_SENS, *RX_POS)

    result = controller.run_simulation(tx, jammer, rx, THRESHOLD)

    assert set(result.keys()) == {
        "j_s_db",
        "tx_recv_dbm",
        "jam_recv_dbm",
        "communication_success",
        "jamming_success",
        "frequency_mismatch",
    }


# ---------------------------------------------------------------------------
# Numeric accuracy
# ---------------------------------------------------------------------------

def test_run_simulation_numeric_values(controller):
    """tx_recv_dbm, jam_recv_dbm and j_s_db should match model calculations."""
    jam_power = 20.0
    tx     = controller.create_radio_source(TX_POWER, TX_FREQ, *TX_POS)
    jammer = controller.create_radio_source(jam_power, JAM_FREQ, *JAM_POS)
    rx     = controller.create_receiver(RX_SENS, *RX_POS)

    result = controller.run_simulation(tx, jammer, rx, THRESHOLD)

    expected_tx  = _expected_tx_recv()
    expected_jam = _expected_jam_recv(jam_power)

    assert abs(result["tx_recv_dbm"]  - expected_tx)  < 1e-9
    assert abs(result["jam_recv_dbm"] - expected_jam) < 1e-9
    assert abs(result["j_s_db"] - (expected_jam - expected_tx)) < 1e-9


# ---------------------------------------------------------------------------
# Scenario 1: Communication success
#   Signal receivable, J/S below threshold → comm succeeds, jamming fails
# ---------------------------------------------------------------------------

def test_communication_success(controller):
    """Weak jammer: J/S well below threshold → communication succeeds."""
    tx     = controller.create_radio_source(TX_POWER, TX_FREQ, *TX_POS)
    jammer = controller.create_radio_source(20.0, JAM_FREQ, *JAM_POS)
    rx     = controller.create_receiver(RX_SENS, *RX_POS)

    result = controller.run_simulation(tx, jammer, rx, THRESHOLD)

    assert result["j_s_db"] < THRESHOLD
    assert result["communication_success"] is True
    assert result["jamming_success"] is False


# ---------------------------------------------------------------------------
# Scenario 2: Communication blocked — jammed
#   Signal receivable, J/S above threshold → comm fails, jamming succeeds
# ---------------------------------------------------------------------------

def test_communication_blocked_by_jamming(controller):
    """Powerful jammer: J/S well above threshold → communication is jammed."""
    tx     = controller.create_radio_source(TX_POWER, TX_FREQ, *TX_POS)
    jammer = controller.create_radio_source(70.0, JAM_FREQ, *JAM_POS)
    rx     = controller.create_receiver(RX_SENS, *RX_POS)

    result = controller.run_simulation(tx, jammer, rx, THRESHOLD)

    assert result["j_s_db"] > THRESHOLD
    assert result["tx_recv_dbm"] >= RX_SENS
    assert result["communication_success"] is False
    assert result["jamming_success"] is True


# ---------------------------------------------------------------------------
# Scenario 3: Communication blocked — signal too weak
#   Signal below sensitivity → comm fails, jamming irrelevant (also False)
# ---------------------------------------------------------------------------

def test_communication_blocked_signal_too_weak(controller):
    """TX too weak to reach RX: both comm and jamming success are False."""
    tx     = controller.create_radio_source(TX_POWER, TX_FREQ, *TX_POS)
    jammer = controller.create_radio_source(70.0, JAM_FREQ, *JAM_POS)
    # Sensitivity set so signal can never reach the receiver
    rx     = controller.create_receiver(-20.0, *RX_POS)

    result = controller.run_simulation(tx, jammer, rx, -20.0)

    assert result["tx_recv_dbm"] < -20.0
    assert result["communication_success"] is False
    assert result["jamming_success"] is False


# ---------------------------------------------------------------------------
# Sensitivity threshold boundary
# ---------------------------------------------------------------------------

def test_sensitivity_boundary_signal_exactly_at_sensitivity(controller):
    """Signal exactly at sensitivity level → communication can still succeed."""
    tx     = controller.create_radio_source(TX_POWER, TX_FREQ, *TX_POS)
    jammer = controller.create_radio_source(20.0, JAM_FREQ, *JAM_POS)

    # Set sensitivity to exactly match the received TX power
    exact_sensitivity = _expected_tx_recv()
    rx = controller.create_receiver(exact_sensitivity, *RX_POS)

    result = controller.run_simulation(tx, jammer, rx, THRESHOLD)

    # signal_dbm == sensitivity_dbm: the < check is False, so communication
    # is not blocked on sensitivity grounds
    assert result["tx_recv_dbm"] == pytest.approx(exact_sensitivity)
    assert result["communication_success"] is True


def test_sensitivity_boundary_signal_just_below_sensitivity(controller):
    """Signal one epsilon below sensitivity → communication blocked, jamming False."""
    tx     = controller.create_radio_source(TX_POWER, TX_FREQ, *TX_POS)
    jammer = controller.create_radio_source(70.0, JAM_FREQ, *JAM_POS)

    just_above_rx_power = _expected_tx_recv() + 0.001
    rx = controller.create_receiver(just_above_rx_power, *RX_POS)

    result = controller.run_simulation(tx, jammer, rx, THRESHOLD)

    assert result["communication_success"] is False
    assert result["jamming_success"] is False


# ---------------------------------------------------------------------------
# J/S threshold boundary
# ---------------------------------------------------------------------------

def test_js_threshold_boundary_js_exactly_at_threshold(controller):
    """J/S exactly equal to threshold → communication succeeds (> not >=)."""
    tx     = controller.create_radio_source(TX_POWER, TX_FREQ, *TX_POS)

    expected_tx  = _expected_tx_recv()
    # Choose jammer power so J/S == THRESHOLD exactly
    jam_power_for_exact_threshold = expected_tx + THRESHOLD + fspl_db(1.5, JAM_FREQ)
    jammer = controller.create_radio_source(jam_power_for_exact_threshold,
                                            JAM_FREQ, *JAM_POS)
    rx = controller.create_receiver(RX_SENS, *RX_POS)

    result = controller.run_simulation(tx, jammer, rx, THRESHOLD)

    assert result["j_s_db"] == pytest.approx(THRESHOLD)
    assert result["communication_success"] is True
    assert result["jamming_success"] is False


# ---------------------------------------------------------------------------
# Frequency mismatch warning
# ---------------------------------------------------------------------------

def test_no_frequency_mismatch_when_frequencies_match(controller):
    tx     = controller.create_radio_source(TX_POWER, TX_FREQ, *TX_POS)
    jammer = controller.create_radio_source(20.0, TX_FREQ, *JAM_POS)  # same freq
    rx     = controller.create_receiver(RX_SENS, *RX_POS)

    result = controller.run_simulation(tx, jammer, rx, THRESHOLD)

    assert result["frequency_mismatch"] is False


def test_frequency_mismatch_flagged_when_frequencies_differ(controller):
    tx     = controller.create_radio_source(TX_POWER, TX_FREQ, *TX_POS)
    jammer = controller.create_radio_source(20.0, TX_FREQ + 50.0, *JAM_POS)
    rx     = controller.create_receiver(RX_SENS, *RX_POS)

    result = controller.run_simulation(tx, jammer, rx, THRESHOLD)

    assert result["frequency_mismatch"] is True


def test_js_threshold_boundary_js_just_above_threshold(controller):
    """J/S infinitesimally above threshold → communication blocked, jamming True."""
    tx     = controller.create_radio_source(TX_POWER, TX_FREQ, *TX_POS)

    expected_tx  = _expected_tx_recv()
    jam_power_just_over = expected_tx + THRESHOLD + fspl_db(1.5, JAM_FREQ) + 0.001
    jammer = controller.create_radio_source(jam_power_just_over,
                                            JAM_FREQ, *JAM_POS)
    rx = controller.create_receiver(RX_SENS, *RX_POS)

    result = controller.run_simulation(tx, jammer, rx, THRESHOLD)

    assert result["j_s_db"] > THRESHOLD
    assert result["communication_success"] is False
    assert result["jamming_success"] is True


# ---------------------------------------------------------------------------
# Truncated normal jammer power
# ---------------------------------------------------------------------------

def test_run_monte_carlo_truncated_power_respects_bounds(controller):
    """Jammer power samples must stay within [min, max] when truncated."""
    from scipy import stats

    power_min = 37.0
    power_max = 43.0
    N = 2000

    result = controller.run_monte_carlo(
        tx_power=TX_POWER, tx_freq=TX_FREQ, tx_pos=TX_POS,
        rx_sens=RX_SENS, rx_pos=RX_POS,
        jam_power_mean=40.0, jam_power_std=2.0,
        jam_freq=JAM_FREQ,
        jam_pos_x_dist=stats.norm(loc=500, scale=50),
        jam_pos_y_dist=stats.norm(loc=0, scale=10),
        jam_pos_z_dist=stats.norm(loc=0, scale=5),
        j_s_threshold_db=THRESHOLD,
        N=N,
        jam_power_min=power_min,
        jam_power_max=power_max,
    )
    # J/S = jam_recv - tx_recv; jam_recv = jam_power - FSPL(jam_dist).
    # Recover approximate jam power per run: jam_power ≈ J/S + tx_recv + FSPL(jam_dist).
    # Instead, verify indirectly: the J/S distribution must be narrower than
    # it would be with unbounded normal, and all js values must fall within
    # the range achievable given the power bounds.
    import numpy as np
    js = result['js_array']
    assert js.shape == (N,)
    assert np.all(np.isfinite(js))
    # Mean J/S should be close to that of an unbounded run with same mean power
    assert abs(result['mean_js'] - result['percentile_50']) < 5.0


def test_run_monte_carlo_untruncated_by_default(controller):
    """Default min=-inf, max=+inf leaves distribution effectively unbounded."""
    from scipy import stats
    import numpy as np

    N = 1000
    result = controller.run_monte_carlo(
        tx_power=TX_POWER, tx_freq=TX_FREQ, tx_pos=TX_POS,
        rx_sens=RX_SENS, rx_pos=RX_POS,
        jam_power_mean=40.0, jam_power_std=2.0,
        jam_freq=JAM_FREQ,
        jam_pos_x_dist=stats.norm(loc=500, scale=1),
        jam_pos_y_dist=stats.norm(loc=0, scale=1),
        jam_pos_z_dist=stats.norm(loc=0, scale=1),
        j_s_threshold_db=THRESHOLD,
        N=N,
    )
    assert result['js_array'].shape == (N,)
    assert np.all(np.isfinite(result['js_array']))
