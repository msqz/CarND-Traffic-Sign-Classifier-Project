#!/usr/bin/env python3
from dataset import DataSet
from lenet import LeNet
from matplotlib import pyplot as plt
from sklearn.utils import shuffle

train = DataSet('./data/train.p')
# train.examples = (train.examples - 128) / train.examples
train.examples, train.labels = shuffle(train.examples, train.labels)
# import pdb; pdb.set_trace()

valid = DataSet('./data/valid.p')
valid.examples = (valid.examples - 128) / valid.examples

network = LeNet(0.001, 100, 128, 0, 0.1)
network.train(train, valid)
accuracy = network.evaluate(valid)

print('Total accuracy: {:.3f}'.format(accuracy))
plt.plot(network.cost_history)
plt.show()
