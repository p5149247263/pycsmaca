from math import floor

from numpy.testing import assert_almost_equal, assert_allclose

from pyqumo.distributions import Constant, Exponential
from pydesim import simulate, Logger
from pycsmaca.simulations import AdHocNetworkModel

# Here we define the parameters those allow us to easily estimate expected
# values manually:
PAYLOAD = 1000  # bits
ACK = 100  # bits
MAC_HEADER = 50  # bits
PHY_HEADER = 25  # bits
PREAMBLE = 1e-3  # seconds
BITRATE = 1e3  # bits per second
DIFS = 200e-3  # 200 ms
SIFS = 100e-3  # 100 ms
SLOT = 50e-3  # 50 ms
CWMIN = 2
CWMAX = 8
SPEED_OF_LIGHT = 1e5   # meters per second
ARRIVAL_MEAN = 2


def test_network_without_collisions():
    radius = 100
    stime_limit = 1000

    # First, we run the simulation:
    sr = simulate(
        AdHocNetworkModel,
        stime_limit=stime_limit,
        params=dict(
            num_stations=2, payload_size=Constant(PAYLOAD), ack_size=ACK,
            mac_header_size=MAC_HEADER, phy_header_size=PHY_HEADER,
            preamble=PREAMBLE, bitrate=BITRATE, difs=DIFS, sifs=SIFS,
            slot=SLOT, cwmin=CWMIN, cwmax=CWMAX, radius=radius,
            speed_of_light=SPEED_OF_LIGHT,
            intervals=Exponential(ARRIVAL_MEAN), queue_capacity=None,
        ),
        loglevel=Logger.Level.WARNING
    )
    assert sr.stime >= stime_limit

    # Since all stations send data to station 0, we assign these stations
    # to more valuable variables:
    access_point = sr.data.stations[0]
    client = sr.data.stations[1]

    # Then we validate that no collisions were registered:
    assert client.receiver.num_collisions == 0
    assert access_point.receiver.num_collisions == 0

    # We also check that average backoff is equal to 0.5 * (CWMIN - 1):
    assert_almost_equal(
        client.transmitter.backoff_vector.mean(),
        0.5 * (CWMIN - 1),
        decimal=1
    )

    # Now we validate that average service time is equal to:
    #   DIFS + SLOT * E[backoff] + delta + PKT + SIFS + delta + ACK,
    # where PKT = (PAYLOAD + MAC_HEADER + PHY_HEADER) / BITRATE + PREAMBLE,
    #       delta = 2 * RADIUS / SPEED_OF_LIGHT (propagation delay):
    packet_duration = (PAYLOAD + MAC_HEADER + PHY_HEADER) / BITRATE + PREAMBLE
    propagation = 2 * radius / SPEED_OF_LIGHT
    mean_backoff = client.transmitter.backoff_vector.mean()
    ack_duration = (ACK + PHY_HEADER) / BITRATE + PREAMBLE
    expected_service_time = (
        DIFS + SLOT * mean_backoff + packet_duration + SIFS +
        2 * propagation + ack_duration
    )
    assert_almost_equal(
        client.transmitter.service_time.mean(),
        expected_service_time,
        decimal=2
    )

    # Check that the number of received packets is proportional to arrival rate:
    expected_num_packets = int(floor(sr.stime / ARRIVAL_MEAN))

    assert access_point.receiver.num_received == client.transmitter.num_sent
    assert_allclose(client.transmitter.num_sent + client.queue.size,
                    expected_num_packets, rtol=0.25)
    # To make sure queue is not overflowing:
    assert 0 < client.queue.size_trace.timeavg() < 20

    # At the end, check that the average packet size is equal to payload:
    assert access_point.sink.packet_sizes.mean() == PAYLOAD
    assert client.source.packet_sizes.mean() == PAYLOAD

    # ... and that interval between generations is as specified by intervals
    # distribution parameter:
    arrival_stats = client.source.arrival_intervals.statistic()
    assert_allclose(arrival_stats.mean(), sr.params.intervals.mean(), rtol=0.1)
    assert_allclose(arrival_stats.std(), sr.params.intervals.std(), rtol=0.25)
