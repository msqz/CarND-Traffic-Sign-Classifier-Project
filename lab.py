#!/usr/bin/env python3
import pickle

with open('./data/train.p', mode='rb') as f:
    data = pickle.load(f)
    X_train, y_train = data['features'], data['labels']

with open('./data/valid.p', mode='rb') as f:
    data = pickle.load(f)
    X_validation, y_validation = data['features'], data['labels']

with open('./data/test.p', mode='rb') as f:
    data = pickle.load(f)
    X_test, y_test = data['features'], data['labels']


assert(len(X_train) == len(y_train))
assert(len(X_validation) == len(y_validation))
assert(len(X_test) == len(y_test))

print()
print("Image Shape: {}".format(X_train[0].shape))
print()
print("Training Set:   {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_validation)))
print("Test Set:       {} samples".format(len(X_test)))


import numpy as np

# Preprocess data
from sklearn.utils import shuffle
import cv2


def gray(data):
    gray = []
    for x in data:
        gray.append(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis])

    return gray


def normalize(data):
    normalized = np.array(data, dtype=np.int16)
    return (normalized - 128) / 128


def replicate(data, labels):
    blurred = []
    for x in data:
        blurred.append(cv2.GaussianBlur(x, (3, 3), 0))

    return np.concatenate((data,
                           np.array(blurred))),\
        np.concatenate((labels, labels))


# X_train = gray(X_train)
X_train, y_train = replicate(X_train, y_train)
X_train = normalize(X_train)
# X_validation = gray(X_validation)
# X_validation = normalize(X_validation)

X_train, y_train = shuffle(X_train, y_train)


# Setup TensorFlow
import tensorflow as tf

EPOCHS = 100
BATCH_SIZE = 128
CHANNELS = 3
CLASSES = 43
LEARNING_RATE = 0.001
BETA = 0.001
DROPOUT = 0.5

# Implement LeNet-5
from tensorflow.contrib.layers import flatten

# Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
mu = 0
sigma = 0.1

weights = {
    'conv1': tf.Variable(tf.truncated_normal([5, 5, CHANNELS, 6], mu, sigma)),
    'conv2': tf.Variable(tf.truncated_normal([5, 5, 6, 16], mu, sigma)),
    'fc1': tf.Variable(tf.truncated_normal([5*5*16, 120], mu, sigma)),
    'fc2': tf.Variable(tf.truncated_normal([120, 84], mu, sigma)),
    'logits': tf.Variable(tf.truncated_normal([84, CLASSES], mu, sigma)),
}

biases = {
    'conv1': tf.Variable(tf.zeros(6)),
    'conv2': tf.Variable(tf.zeros(16)),
    'fc1': tf.Variable(tf.zeros(120)),
    'fc2': tf.Variable(tf.zeros(84)),
    'logits': tf.Variable(tf.zeros(CLASSES)),
}


def LeNet(x, keep_prob):
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1 = tf.nn.conv2d(x, weights['conv1'], [1, 1, 1, 1], 'VALID')
    conv1 = tf.nn.bias_add(conv1, biases['conv1'])

    # TODO: Activation.
    conv1 = tf.nn.relu(conv1)

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    conv2 = tf.nn.conv2d(conv1, weights['conv2'], [1, 1, 1, 1], 'VALID')
    conv2 = tf.nn.bias_add(conv2, biases['conv2'])

    # TODO: Activation.
    conv2 = tf.nn.relu(conv2)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    flat = tf.reshape(conv2, [-1, 400])

    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1 = tf.add(tf.matmul(flat, weights['fc1']), biases['fc1'])

    # TODO: Activation.
    fc1 = tf.nn.relu(fc1)

    fc1 = tf.nn.dropout(fc1, keep_prob)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2 = tf.add(tf.matmul(fc1, weights['fc2']), biases['fc2'])

    # TODO: Activation.
    fc2 = tf.nn.relu(fc2)

    fc2 = tf.nn.dropout(fc2, keep_prob)

    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    logits = tf.add(tf.matmul(fc2, weights['logits']), biases['logits'])

    return logits


# Features and Labels
x = tf.placeholder(tf.float32, (None, 32, 32, CHANNELS))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, CLASSES)
keep_prob = tf.placeholder(tf.float32)


# Training pipeline
rate = LEARNING_RATE

logits = LeNet(x, keep_prob)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
regularizers = tf.nn.l2_loss(weights['fc1']) +\
    tf.nn.l2_loss(weights['fc2']) +\
    tf.nn.l2_loss(weights['logits'])
loss_operation = tf.reduce_mean(loss_operation + BETA * regularizers)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation)


# Model Evaluation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

loss_history = []


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset +
                                  BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={
            x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


# Train the Model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        avg_loss = 0
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            _, loss = sess.run([training_operation, loss_operation],
                               feed_dict={x: batch_x, y: batch_y, keep_prob: DROPOUT})
            avg_loss += loss

        loss_history.append(avg_loss/BATCH_SIZE)
        train_accuracy = evaluate(X_train, y_train)
        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Training Accuracy = {:.3f}".format(train_accuracy))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))

    # saver.save(sess, './lenet')
    # print("Model saved")

from matplotlib import pyplot as plt

plt.plot(loss_history)
plt.show()


# Evaluate the Model
# with tf.Session() as sess:
#     saver.restore(sess, tf.train.latest_checkpoint('.'))

#     test_accuracy = evaluate(X_test, y_test)
#     print("Test Accuracy = {:.3f}".format(test_accuracy))
