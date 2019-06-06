from collections import namedtuple

from .model import SaturatedNetworkModel
from pydesim import simulate, Logger


def adhoc_saturated_network(
        num_clients, payload_size, ack_size, mac_header_size, phy_header_size,
        preamble, bitrate, difs, sifs, slot, cwmin, cwmax, radius=100,
        c=299792458.0, sim_time_limit=100, llevel=Logger.Level.INFO):
    ret = simulate(
        SaturatedNetworkModel,
        stime_limit=sim_time_limit,
        params=dict(
            num_stations=(num_clients + 1), payload_size=payload_size,
            ack_size=ack_size, mac_header_size=mac_header_size,
            phy_header_size=phy_header_size, preamble=preamble, bitrate=bitrate,
            difs=difs, sifs=sifs, slot=slot, cwmin=cwmin, cwmax=cwmax,
            radius=radius, speed_of_light=c
        ), loglevel=llevel
    )

    simret = namedtuple('SimRet', ['clients'])
    client = namedtuple('Client', ['service_time'])

    clients = [client(ret.data.stations[i].transmitter.service_time)
               for i in range(1, num_clients + 1)]

    return simret(clients)