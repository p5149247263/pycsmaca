import collections
import math
from enum import Enum

import numpy as np

from pydesim import Model, Statistic, Trace, Intervals


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
    """Represents radio level of the wireless adapter.

    Parameters:

    - `position`: a 2-D tuple with radio module antenna coordinates.


    Event handlers:

    - `on_frame_rx_begin(frame)`: fired when the frame first symbol reaches the
        radio of this wireless adapter. Informs Receiver about this by calling
        `on_rx_begin(frame.packet)`.

    - `on_frame_rx_end(frame)`: fired when the frame is received completely.
        Radio doesn't check whether the frame was received successfully, it
        just sends the packet from the frame up to the receiver by calling
        `on_rx_end(frame.packet)`.

    - `on_frame_tx_end()`: called when the TX operation is finished. Radio
        informs both Transmitter and Receiver about this by calling
        `on_tx_end()` (without scheduling, using direct call).


    Methods:

    - `transmit(packet)`: schedules `on_frame_rx_begin()` on all peers and
        schedules `on_frame_tx_end()`. When scheduling `on_frame_rx_begin()`,
        it computes the propagation delay based on the distance between this
        radio module and the peer using `position` property.
    """
    def __init__(self, sim, position=(0, 0)):
        super().__init__(sim)
        self._position = np.asarray(position)
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
        return self.connections['receiver'].module

    @property
    def transmitter(self):
        return self.connections['transmitter'].module

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


class ChannelState(Model):
    """Stores the channel state and informs transmitter about updates.
    """
    def __init__(self, sim):
        super().__init__(sim)
        self._is_busy = False

    @property
    def transmitter(self):
        return self.connections['transmitter'].module

    def set_ready(self):
        self._is_busy = False
        self.transmitter.on_channel_ready()

    def set_busy(self):
        self._is_busy = True
        self.transmitter.on_channel_busy()

    @property
    def is_busy(self):
        return self._is_busy

    def __str__(self):
        return f'{self.parent}.channel'


class Transmitter(Model):
    """Models transmitter module at MAC layer."""
    class State(Enum):
        IDLE = 0
        BUSY = 1
        BACKOFF = 2
        TX = 3
        WAIT_ACK = 4

    def __init__(self, sim, address):
        super().__init__(sim)
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
        return self.connections['channel'].module

    @property
    def radio(self):
        return self.connections['radio'].module

    @property
    def queue(self):
        return self.connections['queue'].module

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
            self.packet = None
            self.state = Transmitter.State.IDLE

            self.num_sent += 1
            self.service_time.append(self.sim.stime - self._start_service_time)
            self._start_service_time = None
            self.num_retries_vector.append(self.num_retries)
            self.num_retries = None

            #
            # IMPORTANT: Informing the queue that we can handle the next packet
            #
            self.sim.schedule(0, self.queue.on_transmitter_ready)

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

    def __init__(self, sim, address):
        super().__init__(sim)
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
        return self.connections['radio'].module

    @property
    def channel(self):
        return self.connections['channel'].module

    @property
    def transmitter(self):
        return self.connections['transmitter'].module

    @property
    def sink(self):
        return self.connections['sink'].module

    def on_rx_begin(self, pkt):
        assert pkt not in self.rxbuf

        if self.state is Receiver.State.IDLE and not self.rxbuf:
            self.state = Receiver.State.RX
            self.channel.set_busy()

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
                    self.channel.set_ready()
                else:
                    raise RuntimeError(f'unsupported packet type {pkt.ptype}')
            else:
                self.state = Receiver.State.IDLE
                self.channel.set_ready()

        elif self.state is Receiver.State.COLLIDED:
            if not self.rxbuf:
                self.state = Receiver.State.IDLE
                self.channel.set_ready()
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
                self.channel.set_busy()
                self.state = Receiver.State.COLLIDED
            else:
                self.state = Receiver.State.IDLE

        elif self.state is Receiver.State.TX2:
            if self.rxbuf:
                self.state = Receiver.State.COLLIDED
            else:
                self.channel.set_ready()
                self.state = Receiver.State.IDLE

        elif self.state is Receiver.State.SEND_ACK:
            payload = self.curpkt.payload
            self.sim.schedule(0, self.sink.on_receive, args=(payload,))
            self.num_received += 1
            self.curpkt = None
            self.channel.set_ready()
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


#############################################################################
# MAC Queues models
#############################################################################
class QueueBase(Model):
    def __init__(self, sim):
        super().__init__(sim)

    @property
    def source(self):
        return self.connections['source'].module

    @property
    def transmitter(self):
        return self.connections['transmitter'].module

    def push(self, payload):
        raise NotImplementedError

    def on_transmitter_ready(self):
        raise NotImplementedError

    def __str__(self):
        return f'{self.parent}.queue'


class SaturatedQueue(QueueBase):
    def __init__(self, sim):
        super().__init__(sim)

    def push(self, payload):
        if self.transmitter.state == Transmitter.State.IDLE:
            self.transmitter.send(payload, 0)
        else:
            assert False

    def on_transmitter_ready(self):
        self.sim.schedule(0, self.source.generate)


class Queue(QueueBase):
    def __init__(self, sim, capacity=None):
        super().__init__(sim)
        assert capacity is None or abs(capacity - int(math.ceil(capacity))) == 0
        self.__capacity = capacity
        self.__data = collections.deque()
        self.num_dropped = 0
        self.size_trace = Trace()

    @property
    def size(self):
        return len(self.__data)

    def push(self, payload):
        self.sim.logger.debug(f'push(), size={self.size}, payload={payload}',
                              src=self)
        if self.transmitter.state == Transmitter.State.IDLE:
            self.sim.logger.debug('... passing to transmitter', src=self)
            self.transmitter.send(payload, 0)
        else:
            if self.__capacity is None or self.size < self.__capacity:
                self.__data.append(payload)
                self.size_trace.record(self.sim.stime, self.size)
            else:
                self.num_dropped += 1

    def on_transmitter_ready(self):
        try:
            self.sim.logger.debug(f'pop(), size={self.size}', src=self)
            payload = self.__data.popleft()
            self.transmitter.send(payload, 0)
            self.size_trace.record(self.sim.stime, self.size)
        except IndexError:
            pass  # Do nothing if queue is empty


#############################################################################
# APP Layer: Sources, Sinks
#############################################################################
class SaturatedSource(Model):
    def __init__(self, sim, address, destination=0):
        super().__init__(sim)
        self.address = address
        self.seqn = 0
        self.num_packets = 0
        self.num_bits = 0
        self.packet_sizes = Statistic()
        self.destination = destination
        # Initialization:
        if self.address != destination:
            self.sim.schedule(0, self.generate)

    @property
    def queue(self):
        return self.connections['queue'].module

    def generate(self):
        self.seqn += 1
        try:
            size = self.sim.params.payload_size()
        except TypeError:
            size = self.sim.params.payload_size
        payload = Payload(size, self.address, self.destination, seqn=self.seqn)
        self.queue.push(payload)
        self.num_packets += 1
        self.num_bits += size
        self.packet_sizes.append(size)
        self.sim.logger.info(
            f'sending {payload.size} bits to {payload.destination}',
            src=self
        )

    def __str__(self):
        return f'{self.parent}.source'


class RandomSource(Model):
    def __init__(self, sim, address, dest, intervals, sizes, active=True):
        super().__init__(sim)
        self.__address = address
        self.__seqn = 0
        self.__dest = dest
        self.__intervals, self.__sizes = intervals, sizes
        self.__active = active
        # Statistics:
        self.num_packets = 0
        self.num_bits = 0
        self.packet_sizes = Statistic()
        self.arrival_intervals = Intervals()
        self.arrival_intervals.record(self.sim.stime)
        # Initialization:
        if self.address != dest and active:
            interval = self.__intervals()
            self.sim.schedule(interval, self.generate)

    @property
    def queue(self):
        return self.connections['queue'].module

    @property
    def address(self):
        return self.__address

    @property
    def destination(self):
        return self.__dest

    @property
    def seqn(self):
        return self.__seqn

    @property
    def intervals_distribution(self):
        return self.__intervals

    @property
    def packet_size_distribution(self):
        return self.__sizes

    @property
    def is_active(self):
        return self.__active

    def generate(self):
        self.__seqn += 1
        size = self.__sizes()
        payload = Payload(size, self.address, self.__dest, seqn=self.seqn)
        self.queue.push(payload)
        self.num_packets += 1
        self.num_bits += size
        self.packet_sizes.append(size)
        self.sim.logger.info(
            f'sending {size} bits to {self.destination} from {self.address}',
            src=self
        )
        self.arrival_intervals.record(self.sim.stime)
        if self.address != self.__dest and self.__active:
            interval = self.__intervals()
            self.sim.schedule(interval, self.generate)

    def __str__(self):
        return f'{self.parent}.source'


class Sink(Model):
    def __init__(self, sim, address):
        super().__init__(sim)
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


#############################################################################
# Stations Models
#############################################################################
class StationBase(Model):
    def __init__(self, sim, address):
        super().__init__(sim)

        self.children['sink'] = Sink(sim, address)
        self.children['transmitter'] = Transmitter(sim, address)
        self.children['receiver'] = Receiver(sim, address)
        self.children['channel'] = ChannelState(sim)
        self.children['radio'] = Radio(sim)

        self.transmitter.connections.update({
            'radio': self.radio,
            'channel': self.channel,
        })
        self.receiver.connections.update({
            'radio': self.radio,
            'channel': self.channel,
            'transmitter': self.transmitter,
            'sink': self.sink,
        })
        self.channel.connections['transmitter'] = self.transmitter
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
    def sink(self):
        return self.children.get('sink')

    def __str__(self):
        return f'station:{self.address}'


class SaturatedStation(StationBase):
    def __init__(self, sim, address):
        super().__init__(sim, address=address)

        self.children['source'] = SaturatedSource(sim, address)
        self.children['queue'] = SaturatedQueue(sim)

        self.source.connections['queue'] = self.queue
        self.queue.connections.update({
            'transmitter': self.transmitter,
            'source': self.source,
        })
        self.transmitter.connections.update({
            'queue': self.queue,
        })

    @property
    def queue(self):
        return self.children.get('queue')

    @property
    def source(self):
        return self.children.get('source')


class HalfDuplexStation(StationBase):
    def __init__(self, sim, address, dest, intervals, sizes, active,
                 queue_capacity=None):
        super().__init__(sim, address=address)

        self.children['source'] = RandomSource(
            sim, address, dest, active=active, intervals=intervals, sizes=sizes)
        self.children['queue'] = Queue(sim, queue_capacity)

        self.source.connections['queue'] = self.queue
        self.queue.connections.update({
            'transmitter': self.transmitter,
            'source': self.source,
        })
        self.transmitter.connections.update({
            'queue': self.queue,
        })

    @property
    def queue(self):
        return self.children.get('queue')

    @property
    def source(self):
        return self.children.get('source')


#############################################################################
# Networks Models
#############################################################################
class SaturatedNetworkModel(Model):
    def __init__(self, sim):
        super().__init__(sim)
        ns = self.sim.params.num_stations
        stations = tuple(SaturatedStation(sim, i) for i in range(ns))
        self.children['stations'] = stations
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

    @property
    def stations(self):
        return self.children.get('stations')


class AdHocNetworkModel(Model):
    def __init__(self, sim):
        super().__init__(sim)
        ns = self.sim.params.num_stations
        stations = tuple(
            HalfDuplexStation(
                sim, i, dest=0, active=(i > 0),
                intervals=sim.params.intervals,
                sizes=sim.params.payload_size,
                queue_capacity=sim.params.queue_capacity
            ) for i in range(ns)
        )
        self.children['stations'] = stations
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

    @property
    def stations(self):
        return self.children.get('stations')
