#!/usr/bin/env python3
from dataset import DataSet


train = DataSet('./data/train.p')
validation = DataSet('./data/valid.p')
test = DataSet('./data/test.p')

X_train, y_train = train.features, train.labels

validation.show_frequency(top_n=5)
