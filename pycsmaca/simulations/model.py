from enum import Enum

import numpy as np

from pyqumo.distributions import Constant
from pydesim import Model, simulate, Logger, Statistic


class PacketType(Enum):
    DATA = 0
    ACK = 1


class Payload:
    def __init__(self, size, source, destination, seqn=-1):
        self._size = size
        self._source = source
        self._destination = destination
        self._seqn = seqn

    @property
    def source(self):
        return self._source

    @property
    def destination(self):
        return self._destination

    @property
    def size(self):
        return self._size

    @property
    def seqn(self):
        return self._seqn

    def __str__(self):
        return f"ADU[{self._source}=>{self._destination}, {self._size}b, " \
            f"SEQN={self.seqn}]"


class Packet:
    def __init__(self, ptype, sender, receiver, phy_header_size):
        self._ptype = ptype
        self._sender = sender
        self._receiver = receiver
        self._phy_header_size = phy_header_size

    @property
    def sender(self):
        return self._sender

    @property
    def receiver(self):
        return self._receiver

    @property
    def phy_header_size(self):
        return self._phy_header_size

    @property
    def ptype(self):
        return self._ptype

    @property
    def size(self):
        raise NotImplementedError()


class DataPacket(Packet):
    def __init__(self, sender, receiver, phy_header_size, mac_header_size,
                 payload):
        super().__init__(PacketType.DATA, sender, receiver, phy_header_size)
        self._mac_header_size = mac_header_size
        self._payload = payload

    @property
    def mac_header_size(self):
        return self._mac_header_size

    @property
    def payload(self):
        return self._payload

    @property
    def size(self):
        return self.phy_header_size + self.mac_header_size + self.payload.size

    def __str__(self):
        return f"MPDU[sender={self.sender}, receiver={self.receiver}, " \
            f"size={self.size}b, payload={self.payload}]"


class Ack(Packet):
    def __init__(self, sender, receiver, phy_header_size, mac_size):
        super().__init__(PacketType.ACK, sender, receiver, phy_header_size)
        self._mac_size = mac_size

    @property
    def size(self):
        return self.phy_header_size + self._mac_size

    def __str__(self):
        return f"ACK[sender={self.sender}, receiver={self.receiver}, " \
            f"size={self.size}b]"


class Frame:
    def __init__(self, packet, preamble, bitrate):
        self._packet = packet
        self._preamble = preamble
        self._bitrate = bitrate

    @property
    def packet(self):
        return self._packet

    @property
    def preamble(self):
        return self._preamble

    @property
    def bitrate(self):
        return self._bitrate

    @property
    def duration(self):
        return self.packet.size / self.bitrate + self.preamble

    def __str__(self):
        return f"Frame[{self.duration:.9f}s, packet={self.packet}]"

    def __repr__(self):
        return str(self)


class Radio(Model):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self._position = np.asarray([0, 0])
        self.peers = []

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        assert len(value) == 2
        self._position = np.asarray(value)

    @property
    def receiver(self):
        return self.connections.get('receiver')

    @property
    def transmitter(self):
        return self.connections.get('transmitter')

    def transmit(self, pkt):
        assert isinstance(pkt, Packet)
        frame = Frame(pkt, self.sim.params.preamble, self.sim.params.bitrate)
        for peer in self.peers:
            distance = np.linalg.norm(self.position - peer.position)
            delay = distance / self.sim.params.speed_of_light
            self.sim.schedule(delay, peer.on_frame_rx_begin, args=(frame,))
        self.sim.schedule(frame.duration, self.on_frame_tx_end)

    def on_frame_rx_begin(self, frame):
        assert isinstance(frame, Frame)
        self.receiver.on_rx_begin(frame.packet)
        self.sim.schedule(frame.duration, self.on_frame_rx_end, args=(frame,))

    def on_frame_rx_end(self, frame):
        assert isinstance(frame, Frame)
        self.receiver.on_rx_end(frame.packet)

    def on_frame_tx_end(self):
        self.transmitter.on_tx_end()
        self.receiver.on_tx_end()

    def __str__(self):
        return f'{self.parent}.radio'


class Channel(Model):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self._is_busy = False

    @property
    def transmitter(self):
        return self.connections.get('transmitter')

    def ready(self):
        self._is_busy = False
        self.transmitter.on_channel_ready()

    def busy(self):
        self._is_busy = True
        self.transmitter.on_channel_busy()

    @property
    def is_busy(self):
        return self._is_busy

    def __str__(self):
        return f'{self.parent}.channel'


class Transmitter(Model):
    class State(Enum):
        IDLE = 0
        BUSY = 1
        BACKOFF = 2
        TX = 3
        WAIT_ACK = 4

    def __init__(self, address, parent=None):
        super().__init__(parent=parent)
        self.timeout = None
        self.cw = 65536
        self.backoff = -1
        self.num_retries = None
        self.packet = None
        self.address = address
        self._state = Transmitter.State.IDLE
        self.backoff_vector = Statistic()
        self._start_service_time = None
        self.service_time = Statistic()
        self.num_sent = 0
        self.num_retries_vector = Statistic()

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        if self.state != state:
            self.sim.logger.debug(
                f'{self.state.name} -> {state.name}', src=self
            )
        self._state = state

    @property
    def channel(self):
        return self.connections.get('channel')

    @property
    def radio(self):
        return self.connections.get('radio')

    @property
    def queue(self):
        return self.connections.get('queue')

    def send(self, payload, rcv_addr):
        assert self.state == Transmitter.State.IDLE

        self.cw = self.sim.params.cwmin
        self.backoff = np.random.randint(0, self.cw)
        self.num_retries = 1
        self.packet = DataPacket(
            self.address, rcv_addr, self.sim.params.phy_header_size,
            self.sim.params.mac_header_size, payload
        )

        self._start_service_time = self.sim.stime
        self.backoff_vector.append(self.backoff)

        self.sim.logger.debug(
            f'backoff := {self.backoff} (CW={self.cw}, NR={self.num_retries})',
            src=self
        )

        if self.channel.is_busy:
            self.state = Transmitter.State.BUSY
        else:
            self.state = Transmitter.State.BACKOFF
            self.timeout = self.sim.schedule(
                self.sim.params.difs, self.on_backoff_timeout
            )

    def on_channel_ready(self):
        if self.state == Transmitter.State.BUSY:
            self.timeout = self.sim.schedule(
                self.sim.params.difs, self.on_backoff_timeout
            )
            self.state = Transmitter.State.BACKOFF

    def on_channel_busy(self):
        if self.state == Transmitter.State.BACKOFF:
            self.sim.cancel(self.timeout)
            self.state = Transmitter.State.BUSY

    def on_tx_end(self):
        if self.state == Transmitter.State.TX:
            self.sim.logger.debug('TX finished', src=self)
            ack_duration = (
                    (self.sim.params.ack_size + self.sim.params.phy_header_size)
                    / self.sim.params.bitrate + self.sim.params.preamble +
                    6 * self.sim.params.radius / self.sim.params.speed_of_light
            )
            self.timeout = self.sim.schedule(
                self.sim.params.sifs + ack_duration, self.on_no_ack)
            self.state = Transmitter.State.WAIT_ACK

    def on_ack(self):
        if self.state == Transmitter.State.WAIT_ACK:
            self.sim.logger.debug('received ACK', src=self)

            self.sim.cancel(self.timeout)
            self.sim.schedule(0, self.queue.on_transmitter_ready)
            self.packet = None
            self.state = Transmitter.State.IDLE

            self.num_sent += 1
            self.service_time.append(self.sim.stime - self._start_service_time)
            self._start_service_time = None
            self.num_retries_vector.append(self.num_retries)
            self.num_retries = None

    def on_no_ack(self):
        assert self.state == Transmitter.State.WAIT_ACK
        self.num_retries += 1
        self.cw = min(2 * self.cw, self.sim.params.cwmax)
        self.backoff = np.random.randint(0, self.cw)

        self.backoff_vector.append(self.backoff)

        self.sim.logger.debug(
            f'backoff := {self.backoff} (CW={self.cw}, NR={self.num_retries})',
            src=self
        )

        if self.channel.is_busy:
            self.state = Transmitter.State.BUSY
        else:
            self.state = Transmitter.State.BACKOFF
            self.timeout = self.sim.schedule(
                self.sim.params.difs, self.on_backoff_timeout
            )

    def on_backoff_timeout(self):
        if self.backoff == 0:
            self.state = Transmitter.State.TX
            self.radio.transmit(self.packet)
        else:
            assert self.backoff > 0
            self.backoff -= 1
            self.timeout = self.sim.schedule(
                self.sim.params.slot, self.on_backoff_timeout
            )
            self.sim.logger.debug(f'backoff := {self.backoff}', src=self)

    def __str__(self):
        return f'{self.parent}.transmitter'


class Receiver(Model):
    class State(Enum):
        IDLE = 0
        RX = 1
        TX1 = 2
        TX2 = 3
        COLLIDED = 4
        WAIT_SEND_ACK = 5
        SEND_ACK = 6

    def __init__(self, address, parent=None):
        super().__init__(parent=parent)
        self._state = Receiver.State.IDLE
        self.rxbuf = set()
        self.curpkt = None
        self.address = address
        self.num_collisions = 0
        self.num_received = 0

    @property
    def state(self):
        return self._state

    @property
    def collision_probability(self):
        if self.num_received > 0 or self.num_collisions > 0:
            return self.num_collisions / (
                    self.num_collisions + self.num_received)
        return 0

    @state.setter
    def state(self, state):
        if state != self.state:
            self.sim.logger.debug(
                f'{self.state.name} -> {state.name}', src=self
            )
            if state is Receiver.State.COLLIDED:
                self.num_collisions += 1
            self._state = state

    @property
    def radio(self):
        return self.connections.get('radio')

    @property
    def channel(self):
        return self.connections.get('channel')

    @property
    def transmitter(self):
        return self.connections.get('transmitter')

    @property
    def sink(self):
        return self.connections.get('sink')

    def on_rx_begin(self, pkt):
        assert pkt not in self.rxbuf

        if self.state is Receiver.State.IDLE and not self.rxbuf:
            self.state = Receiver.State.RX
            self.channel.busy()

        elif (self.state is Receiver.State.RX or (
                self.state is Receiver.State.IDLE and self.rxbuf)):
            self.state = Receiver.State.COLLIDED

        self.rxbuf.add(pkt)

        # In all other states (e.g. TX2, WAIT_SEND_ACK, SEND_ACK) we just add
        # the packet to RX buffer.

    def on_rx_end(self, pkt):
        assert pkt in self.rxbuf
        self.rxbuf.remove(pkt)

        if self.state is Receiver.State.RX:
            assert not self.rxbuf
            if pkt.receiver == self.address:
                if pkt.ptype is PacketType.DATA:
                    self.state = Receiver.State.WAIT_SEND_ACK
                    self.curpkt = pkt
                    self.sim.schedule(self.sim.params.sifs, self.on_timeout)
                elif pkt.ptype is PacketType.ACK:
                    self.transmitter.on_ack()
                    self.state = Receiver.State.IDLE
                    self.channel.ready()
                else:
                    raise RuntimeError(f'unsupported packet type {pkt.ptype}')
            else:
                self.state = Receiver.State.IDLE
                self.channel.ready()

        elif self.state is Receiver.State.COLLIDED:
            if not self.rxbuf:
                self.state = Receiver.State.IDLE
                self.channel.ready()
            # Otherwise stay in COLLIDED state

        # In all other states (e.g. IDLE, TX1, TX2, ...) we just purge the
        # packet from the RX buffer.

    def on_tx_begin(self):
        if self.state is Receiver.State.IDLE:
            self.state = Receiver.State.TX1

        elif self.state in (Receiver.State.RX, Receiver.State.COLLIDED):
            self.state = Receiver.State.TX2

        assert self.state is not self.state.WAIT_SEND_ACK

    def on_tx_end(self):
        if self.state is Receiver.State.TX1:
            if self.rxbuf:
                self.channel.busy()
                self.state = Receiver.State.COLLIDED
            else:
                self.state = Receiver.State.IDLE

        elif self.state is Receiver.State.TX2:
            if self.rxbuf:
                self.state = Receiver.State.COLLIDED
            else:
                self.channel.ready()
                self.state = Receiver.State.IDLE

        elif self.state is Receiver.State.SEND_ACK:
            payload = self.curpkt.payload
            self.sim.schedule(0, self.sink.on_receive, args=(payload,))
            self.num_received += 1
            self.curpkt = None
            self.channel.ready()
            self.state = Receiver.State.IDLE

    def on_timeout(self):
        assert self.state is Receiver.State.WAIT_SEND_ACK
        ack = Ack(
            self.address, self.curpkt.sender, self.sim.params.phy_header_size,
            self.sim.params.ack_size
        )
        self.radio.transmit(ack)
        self.state = Receiver.State.SEND_ACK

    def __str__(self):
        return f'{self.parent}.receiver'


class Queue(Model):
    def __init__(self, parent=None):
        super().__init__(parent=parent)

    @property
    def source(self):
        return self.connections.get('source')

    @property
    def transmitter(self):
        return self.connections.get('transmitter')

    def push(self, payload):
        if self.transmitter.state == Transmitter.State.IDLE:
            self.transmitter.send(payload, 0)
        else:
            assert False

    def on_transmitter_ready(self):
        self.sim.schedule(0, self.source.generate)

    def __str__(self):
        return f'{self.parent}.queue'


class Source(Model):
    def __init__(self, address, parent=None):
        super().__init__(parent=parent)
        self.address = address
        self.seqn = 0
        self.num_packets = 0
        self.num_bits = 0
        self.packet_sizes = Statistic()

    @property
    def queue(self):
        return self.connections.get('queue')

    def generate(self):
        self.seqn += 1
        try:
            size = self.sim.params.payload_size()
        except TypeError:
            size = self.sim.params.payload_size
        payload = Payload(size, self.address, 0, seqn=self.seqn)
        self.queue.push(payload)
        self.num_packets += 1
        self.num_bits += size
        self.packet_sizes.append(size)
        self.sim.logger.info(
            f'sending {payload.size} bits to {payload.destination}',
            src=self
        )

    def initialize(self):
        if self.address != 0:
            self.sim.schedule(0, self.generate)

    def __str__(self):
        return f'{self.parent}.source'


class Sink(Model):
    def __init__(self, address, parent=None):
        super().__init__(parent=parent)
        self.dsn = {}
        self.address = address
        self.num_packets = 0
        self.num_bits = 0
        self.packet_sizes = Statistic()

    def on_receive(self, payload):
        self.sim.logger.info(
            f'received {payload.size} bits from {payload.source}',
            src=self
        )
        self.num_packets += 1
        self.num_bits += payload.size
        self.packet_sizes.append(payload.size)

    def __str__(self):
        return f'{self.parent}.sink'


class Station(Model):
    def __init__(self, address, parent=None):
        super().__init__(parent=parent)

        self.children.add('source', Source(address, parent=self))
        self.children.add('sink', Sink(address, parent=self))
        self.children.add('queue', Queue(parent=self))
        self.children.add('transmitter', Transmitter(address, parent=self))
        self.children.add('receiver', Receiver(address, parent=self))
        self.children.add('channel', Channel(parent=self))
        self.children.add('radio', Radio(parent=self))

        self.source.connections.add('queue', self.queue)
        self.queue.connections.update({
            'transmitter': self.transmitter,
            'source': self.source,
        })
        self.transmitter.connections.update({
            'radio': self.radio,
            'channel': self.channel,
            'queue': self.queue,
        })
        self.receiver.connections.update({
            'radio': self.radio,
            'channel': self.channel,
            'transmitter': self.transmitter,
            'sink': self.sink,
        })
        self.channel.connections.add('transmitter', self.transmitter)
        self.radio.connections.update({
            'receiver': self.receiver,
            'transmitter': self.transmitter,
        })

        self.address = address

    @property
    def radio(self):
        return self.children.get('radio')

    @property
    def channel(self):
        return self.children.get('channel')

    @property
    def transmitter(self):
        return self.children.get('transmitter')

    @property
    def receiver(self):
        return self.children.get('receiver')

    @property
    def queue(self):
        return self.children.get('queue')

    @property
    def source(self):
        return self.children.get('source')

    @property
    def sink(self):
        return self.children.get('sink')

    def __str__(self):
        return f'station:{self.address}'


class SaturatedNetworkModel(Model):
    def __init__(self):
        super().__init__()

    @property
    def stations(self):
        return self.children.get('stations')

    def initialize(self):
        ns = self.sim.params.num_stations
        stations = tuple(Station(i, parent=self) for i in range(ns))
        self.children.add('stations', stations)

        for i in range(ns):
            station = stations[i]
            radio = station.children.get('radio')
            station.children.get('radio').position = (
                self.sim.params.radius * np.cos(2 * np.pi / ns * i),
                self.sim.params.radius * np.sin(2 * np.pi / ns * i),
            )
            for j in range(self.sim.params.num_stations):
                if i == j:
                    continue
                radio.peers.append(stations[j].children.get('radio'))


if __name__ == '__main__':
    ret = simulate(
        SaturatedNetworkModel,
        stime_limit=10000,
        params=dict(
            num_stations=2,
            payload_size=Constant(1000),
            ack_size=100,
            mac_header_size=50,
            phy_header_size=25,
            preamble=10e-3,
            bitrate=1000,
            difs=200e-3,
            sifs=100e-3,
            slot=50e-3,
            cwmin=4,
            cwmax=64,
            radius=100 / np.sqrt(3),
            speed_of_light=1e5
        ),
        loglevel=Logger.Level.WARNING
    )
    print('collision probability: ',
          ret.data.stations[0].receiver.collision_probability)
    print('service duration: ',
          ret.data.stations[1].transmitter.service_time.mean())
    print('throughput: ',
          ret.data.stations[0].sink.num_bits / ret.stime)

# def __init__(self, ):
#     delay = delay if delay is not None else slot / 100
#     self.parameters = namedtuple('SaturatedNetworkParameters', [
#         'num_stations', 'payload_size', 'ack_size',
#         'mac_header_size', 'phy_header_size', 'preamble', 'bitrate',
#         'difs', 'sifs', 'slot', 'cwmin', 'cwmax', 'delay',
#     ])(num_stations, payload_size, ack_size,
#        mac_header_size, phy_header_size, preamble, bitrate,
#        difs, sifs, slot, cwmin, cwmax, delay)
