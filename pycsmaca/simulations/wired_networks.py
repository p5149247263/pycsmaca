from pydesim import Model

from pycsmaca.simulations.modules import RandomSource, WiredTransceiver, Queue, \
    WiredInterface
from .station import Station


class WiredLineNetwork(Model):
    def __init__(self, sim):
        super().__init__(sim)

        if sim.params.num_stations < 2:
            raise ValueError('minimum number of stations in network is 2')

        # Reading parameters and building stations:
        stations = []
        next_address, n = 1, sim.params.num_stations
        destination_address = 2 + (n - 2) * 2
        for i in range(n):
            if i in sim.params.active_sources:
                source = RandomSource(
                    sim, sim.params.payload_size, sim.params.source_interval,
                    source_id=i, dest_addr=destination_address
                )
            else:
                source = None

            # Building wired interfaces:
            interfaces = []
            for _ in range(2 if 0 < i < n - 1 else 1):
                transceiver = WiredTransceiver(
                    sim, bitrate=sim.params.bitrate,
                    header_size=sim.params.header_size,
                    preamble=sim.params.preamble,
                    ifs=sim.params.ifs,
                )
                queue = Queue(sim)
                iface = WiredInterface(sim, next_address, queue, transceiver)
                next_address += 1
                interfaces.append(iface)

            # Building station:
            sta = Station(sim, source=source, interfaces=interfaces)
            stations.append(sta)

            # Writing routing table for all station except the last one:
            if i < n - 1:
                out_iface = interfaces[-1]
                switch_conn = sta.get_switch_connection_for(out_iface)
                sta.switch.table.add(
                    destination_address,
                    switch_conn.name,
                    out_iface.address + 1
                )

        # Adding stations as children:
        self.children['stations'] = stations

        # Connecting stations interfaces in chain:
        for i in range(n - 1):
            if1, if2 = stations[i].interfaces[-1], stations[i+1].interfaces[0]
            conn = if1.connections.set('wire', if2, rname='wire')
            conn.delay = sim.params.distance / sim.params.speed_of_light

    @property
    def stations(self):
        return self.children['stations']

    @property
    def num_stations(self):
        return len(self.stations)

    def get_tx_iface(self, sta_index):
        if sta_index < self.num_stations - 1:
            return self.stations[sta_index].interfaces[-1]
        raise ValueError(
            f'TX interface defined for the first {self.num_stations - 1} '
            f'stations only')

    def get_rx_iface(self, sta_index):
        if sta_index > 0:
            return self.stations[sta_index].interfaces[0]
        raise ValueError(
            f'RX interface defined for the last {self.num_stations - 1} '
            f'stations only')

    def __str__(self):
        return 'Network'

    def print_children(self):
        def get_all_leafs(module):
            children = [module]
            for child in module.children.all():
                children.extend(get_all_leafs(child))
            return children

        print('Network components:')
        for m in get_all_leafs(self):
            print(f'+ {m}')
