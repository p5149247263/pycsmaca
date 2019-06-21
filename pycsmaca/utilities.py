SPEED_OF_LIGHT = 299792458.0


class ReadOnlyDict:
    def __init__(self, data):
        self.__data = data

    def __getitem__(self, item):
        return self.__data[item]

    def __contains__(self, item):
        return item in self.__data

    def items(self):
        return self.__data.items()

    def values(self):
        return self.__data.values()

    def keys(self):
        return self.__data.keys()

    def __eq__(self, other):
        try:
            return self.__data == other.__data
        except AttributeError:
            return self.__data == other

    def copy(self):
        return ReadOnlyDict(self.__data.copy())

    def __iter__(self):
        return iter(self.__data)

    def __str__(self):
        return 'RODict' + str(self.__data)
