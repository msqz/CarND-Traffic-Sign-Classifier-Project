import pickle
import csv
from matplotlib import pyplot as plt
import numpy as np
from operator import itemgetter


class DataSet:
    def __init__(self, fp):
        with open(fp, mode='rb') as f:
            self.data = pickle.load(f)

        self.features = self.data['features']
        self.labels = self.data['labels']

        self.classes = {}
        with open('./signnames.csv', newline='') as f:
            reader = csv.reader(f)
            unique = set(['{}'.format(l) for l in self.labels])
            for row in reader:
                if row[0] in unique:
                    self.classes[row[0]] = row[1]

        self.n_examples = len(self.features)
        self.n_classes = len(self.classes)
        self.image_shape = self.features[0].shape

    def show_example(self, i):
        plt.imshow(self.features[i])
        plt.title(self.classes['{}'.format(self.labels[i])])
        plt.show()

    def show_frequency(self, top_n=None):
        fig, ax = plt.subplots()

        hist, bins = np.histogram(
            self.labels, bins=range(self.n_classes + 1))
        ticks = range(self.n_classes)
        labels = self.classes.keys()
        title = 'Frequency'

        if top_n is not None:
            ranked = sorted(
                list(zip(hist, bins)),
                key=itemgetter(0),
                reverse=True)

            hist = [r[0] for r in ranked][:top_n]
            bins = range(len(hist) + 1)
            ticks = range(len(bins))
            labels = [r[1] for r in ranked][:top_n]
            title = 'Top {} frequency ranking'.format(top_n)

        ax.bar(bins[:-1], hist)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        plt.show()
