import os
import matplotlib.pyplot as plt


class Plot:

    @staticmethod
    def plot(*args, **kwargs):
        return NotImplementedError


class Data:

    def __init__(self):
        pass

    def add(self, identifier, value):
        pass


class TextData(Data):

    def __init__(self):
        super().__init__()
        self._values = {}

    def __repr__(self):
        text = ''
        for key in self._values:
            text += f'\n{key}:'
            for log in self._values[key]:
                for line in log:
                    text += line
                text += '\n'
        return text

    def add(self, identifier, value):
        if self._values.get(identifier, None) is None:
            self._values[identifier] = []
        self._values[identifier].append(value)


class ScalarData(Data, Plot):

    @staticmethod
    def summed_average(scalar_iterable, identifiers=None):
        return (sum([scalar.average(identifiers) for scalar in scalar_iterable])
                / len(scalar_iterable))

    # noinspection PyMethodOverriding
    @staticmethod
    def plot(x, y, plot_size):
        plt.plot()

    def __init__(self):
        super().__init__()
        self._values = {}

    def add(self, identifier, value):
        if self._values.get(identifier, None) is None:
            self._values[identifier] = [0, 0]
        self._values[identifier][0] += value
        self._values[identifier][1] += 1

    def average(self, identifiers=None):
        if identifiers is None:
            identifiers = self._values.keys()
        if len(self._values) == 0:
            return 0
        return (sum([self._values[key][0]/self._values[key][1] for key in self._values if key in identifiers])
                / len(self._values))


class LatentStateData(Data, Plot):

    def __init__(self):
        super().__init__()

    # noinspection PyMethodOverriding
    @staticmethod
    def plot(x, y, plot_size):
        plt.plot()


class AttentionData(Data, Plot):

    def __init__(self):
        super().__init__()

    # noinspection PyMethodOverriding
    @staticmethod
    def plot(x, y, plot_size):
        plt.plot()


class DataLog:

    def __init__(self, data_interface):
        self._data = dict(zip(list(data_interface.keys()), [data_type() for data_type in data_interface.values()]))

    # noinspection PyUnresolvedReferences
    def add(self, identifier, key, value):
        self._data[key].add(identifier, value)

    @property
    def data(self):
        return self._data


class Analyzer:

    def __init__(self, directory):
        self._directory = directory
        self._data_profile = None

    def _collect_data(self):
        files = sorted(os.listdir(self._directory), key=lambda x: x.split('_')[1])
        if len(files) == 0:
            return None
        else:
            return None

    def show_available_metrics(self):
        self._data_profile = None
        print('alignment_weights, hidden_states')

    def add_metric(self, metric, epoch_range):
        pass

    def plot(self, metric):
        plt.plot([1, 2, 3, 4])
        plt.show()
