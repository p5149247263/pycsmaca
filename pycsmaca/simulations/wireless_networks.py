from pydesim import Model

from pycsmaca.simulations.modules import RandomSource, Queue, Transmitter, \
    Receiver, Radio, ConnectionManager, WirelessInterface
from .station import Station


class WirelessHalfDuplexLineNetwork(Model):
    def __init__(self, sim):
        super().__init__(sim)

        if sim.params.num_stations < 2:
            raise ValueError('minimum number of stations in network is 2')

        # Building connection manager:
        conn_manager = ConnectionManager(sim)

        # Reading parameters and building stations:
        stations = []
        n = sim.params.num_stations
        destination_address = n
        for i in range(n):
            if i in sim.params.active_sources:
                source = RandomSource(
                    sim, sim.params.payload_size, sim.params.source_interval,
                    source_id=i, dest_addr=destination_address
                )
            else:
                source = None

            # Building wireless interfaces:
            conn_radius = sim.params.connection_radius
            max_propagation = conn_radius / sim.params.speed_of_light
            transmitter = Transmitter(sim, max_propagation=max_propagation)
            receiver = Receiver(sim)
            queue = Queue(sim)
            radio = Radio(sim, conn_manager, connection_radius=conn_radius,
                          position=(i * sim.params.distance, 0))
            iface = WirelessInterface(sim, i + 1, queue, transmitter,
                                      receiver, radio)

            # Building station:
            sta = Station(sim, source=source, interfaces=[iface])
            stations.append(sta)

            # Writing routing table for all station except the last one:
            if i < n - 1:
                switch_conn = sta.get_switch_connection_for(iface)
                sta.switch.table.add(
                    destination_address,
                    switch_conn.name,
                    iface.address + 1
                )

        # Adding stations as children:
        self.children['stations'] = stations

    @property
    def stations(self):
        return self.children['stations']

    @property
    def num_stations(self):
        return len(self.stations)

    def get_iface(self, sta_index):
        if sta_index < self.num_stations:
            return self.stations[sta_index].interfaces[-1]
        raise ValueError(f'station index {sta_index} out of bounds')

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
