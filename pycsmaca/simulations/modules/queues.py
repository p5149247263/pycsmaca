from collections import deque

from pydesim import Model, Trace


class Queue(Model):
    """`Queue` model stores packets and forwards them to services by requests.

    `Queue` module may have finite or infinite capacity, which is specified
    during the `Queue` instance creation. If the capacity is finite, then
    packets pushed when the queue is full will be dropped. In other case,
    the queue will store all the packets being received.

    `Queue` can support multiple services (e.g., network interfaces). It
    delivers packets when a service calls `get_next(service)` method on
    the queue.

    > IMPORTANT: to use a queue, service MUST be connected using
    bi-directional connection.

    If multiple services request packets, they are delivered in the order of
    `get_next()` was called. Each call leads to a single packet being
    transmitted.

    The packet is transmitted to the service immediately after `get_next()`
    call, if the queue is not empty. If the queue is empty, it stores a request
    and send a packet right after a new packet arrives. In the latter case,
    queue size is not updated, and it is counted as the packet went directly
    to the service.

    Packets are sent to the services via `connection.send()` call, and this
    is the reason why the service is meant to be connected to the queue.

    `Queue` accepts packets via its `handle_message()` call.

    Connections:

    - any service that is going to use a queue as a data source MUST be
        connected (these connections can have any names);

    - data producers should also be connected to use `connection.send()` call
        that causes `handle_message(packet)` being called (these connections
        can also have any names).

    > NOTE: queue can also be accessed with `push()` and `pop()` methods,
    so in some cases producers and services may use this module directly,
    without connections (and without `get_next()` also in this case).
    """
    def __init__(self, sim, capacity=None):
        super().__init__(sim)
        self.__capacity = capacity
        self.__packets = deque()
        self.__num_dropped = 0
        self.__data_requests = deque()
        # Statistics:
        self.__size_trace = Trace()
        self.__bitsize_trace = Trace()
        self.__size_trace.record(sim.stime, 0)
        self.__bitsize_trace.record(sim.stime, 0)

    @property
    def capacity(self):
        return self.__capacity

    @property
    def num_dropped(self):
        return self.__num_dropped

    @property
    def size_trace(self):
        return self.__size_trace

    @property
    def bitsize_trace(self):
        return self.__bitsize_trace

    def empty(self):
        return len(self) == 0

    def full(self):
        return len(self) == self.capacity

    def __len__(self):
        return len(self.__packets)

    def size(self):
        return len(self)

    def bitsize(self):
        return sum(pkt.size for pkt in self.__packets)

    def as_tuple(self):
        return tuple(self.__packets)

    def push(self, packet):
        if self.__data_requests:
            connection = self.__data_requests.popleft()
            connection.send(packet)
        else:
            if self.capacity is None or len(self) < self.capacity:
                self.__packets.append(packet)
                self.__size_trace.record(self.sim.stime, len(self))
                self.__bitsize_trace.record(self.sim.stime, self.bitsize())
            else:
                self.__num_dropped += 1

    def pop(self):
        try:
            ret = self.__packets.popleft()
        except IndexError as err:
            raise ValueError('pop from empty Queue') from err
        else:
            self.__size_trace.record(self.sim.stime, len(self))
            self.__bitsize_trace.record(self.sim.stime, self.bitsize())
            return ret

    def get_next(self, service):
        connection = self._get_connection_to(service)
        if not self.empty():
            connection.send(self.pop())
        else:
            self.__data_requests.append(connection)

    def handle_message(self, message, connection=None, sender=None):
        self.push(message)

    def _get_connection_to(self, module):
        for conn_name, peer in self.connections.as_dict().items():
            if module == peer:
                return self.connections[conn_name]
        raise ValueError(f'connection to {module} not found')

    def __str__(self):
        prefix = f'{self.parent}.' if self.parent is not None else ''
        return f'{prefix}Queue'