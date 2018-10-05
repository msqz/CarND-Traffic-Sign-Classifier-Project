#!/usr/bin/env python3
from dataset import DataSet
from lenet import LeNet

train = DataSet('./data/train.p')
valid = DataSet('./data/valid.p')

print(train.image_shape)

network = LeNet(0.001, 10, 128, 0, 0.1)
network.train(train)
accuracy = network.evaluate(valid)
print('Total accuracy: {:.3f}'.format(accuracy))
