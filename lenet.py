import tensorflow as tf


class LeNet:
    def __init__(self, learning_rate, epochs, batch_size, mu, sigma):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights = {
            'conv1': tf.Variable(tf.truncated_normal([5, 5, 3, 6], mu, sigma)),
            'conv2': tf.Variable(
                tf.truncated_normal([5, 5, 6, 16], mu, sigma)),
            'fc1': tf.Variable(tf.truncated_normal([5*5*16, 120], mu, sigma)),
            'fc2': tf.Variable(tf.truncated_normal([120, 84], mu, sigma)),
            'out': tf.Variable(tf.truncated_normal([84, 10], mu, sigma)),
        }

        self.biases = {
            'conv1': tf.Variable(tf.zeros([6])),
            'conv2': tf.Variable(tf.zeros([16])),
            'fc1': tf.Variable(tf.zeros([120])),
            'fc2': tf.Variable(tf.zeros([84])),
            'out': tf.Variable(tf.zeros([10])),
        }

    def run(self, x):
        conv1 = tf.nn.conv2d(x, self.weights['conv1'], [1, 1, 1, 1], 'VALID')
        conv1 = tf.nn.bias_add(conv1, self.biases['conv1'])
        conv1 = tf.nn.relu(conv1)
        conv1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

        conv2 = tf.nn.conv2d(
            conv1, self.weights['conv2'], [1, 1, 1, 1], 'VALID')
        conv2 = tf.nn.bias_add(conv2, self.biases['conv2'])
        conv2 = tf.nn.relu(conv2)
        conv2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

        flat = tf.reshape(conv2, [-1, 400])
        fc1 = tf.add(tf.matmul(flat, self.weights['fc1']), self.biases['fc1'])
        fc1 = tf.nn.relu(fc1)

        fc2 = tf.add(tf.matmul(fc1, self.weights['fc2']), self.biases['fc2'])
        fc2 = tf.nn.relu(fc2)

        out = tf.add(tf.matmul(fc2, self.weights['out']), self.biases['out'])

        return out

    def train(self, dataset):
        x = tf.placeholder(tf.float32, (None, 32, 32, 3))
        y = tf.placeholder(tf.int32, (None))
        one_hot = tf.one_hot(y, 10)

        logits = self.run(x)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=one_hot, logits=logits)
        cost = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        training_op = optimizer.minimize(cost)

        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            for e in range(self.epochs):
                for offset in range(0, dataset.n_examples, self.batch_size):
                    end = offset + self.batch_size
                    batch_x = dataset.examples[offset:end]
                    batch_y = dataset.labels[offset:end]
                    session.run(training_op, feed_dict={
                                x: batch_x, y: batch_y})

    def evaluate(self, dataset):
        x = tf.placeholder(tf.float32, (None, 32, 32, 3))
        y = tf.placeholder(tf.int32, (None))

        logits = self.run(x)
        one_hot = tf.one_hot(y, 10)
        correct_prediction = tf.equal(tf.argmax(logits), tf.argmax(one_hot))
        accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        total_accuracy = 0
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            for offset in range(0, dataset.n_examples, self.batch_size):
                end = offset + self.batch_size
                batch_x = dataset.examples[offset:end]
                batch_y = dataset.labels[offset:end]
                accuracy = session.run(
                    accuracy_op, feed_dict={x: batch_x, y: batch_y})
                total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / dataset.n_examples

    def classify(X, y):
        pass

    def save(self):
        pass

    def load(self):
        pass
