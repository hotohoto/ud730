from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

pickle_file = 'notMNIST.pickle'
seed = 19

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

batch_size = 128
logs_path = "/tmp/ud730"

# n_hidden_nodes = [20]
n_hidden_nodes = [1024]
n_all_nodes = [image_size * image_size, *n_hidden_nodes, num_labels]

graph = tf.Graph()
with graph.as_default():
    with tf.name_scope('input'):
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

    # Variables.

    weights = [None] * (len(n_hidden_nodes) + 1)

    biases = [None] * (len(n_hidden_nodes) + 1)

    train_net_layer_out = [None] * (len(n_hidden_nodes) + 1)
    valid_net_layer_out = [None] * (len(n_hidden_nodes) + 1)
    test_net_layer_out = [None] * (len(n_hidden_nodes) + 1)

    for i in range(len(n_hidden_nodes) + 1):
        print(i, n_all_nodes[i], n_all_nodes[i + 1])
        with tf.name_scope('weights' + str(i)):
            weights[i] = tf.Variable(tf.truncated_normal([n_all_nodes[i], n_all_nodes[i + 1]], seed=seed))
        with tf.name_scope('biases' + str(i)):
            biases[i] = tf.Variable(tf.zeros([n_all_nodes[i + 1]]))
        with tf.name_scope('layers' + str(i)):
            if i == 0:
                train_net_layer_out[0] = tf.nn.relu(tf.matmul(tf_train_dataset, weights[0]) + biases[0])
                valid_net_layer_out[0] = tf.nn.relu(tf.matmul(tf_valid_dataset, weights[0]) + biases[0])
                test_net_layer_out[0] = tf.nn.relu(tf.matmul(tf_test_dataset, weights[0]) + biases[0])
            elif i < len(n_hidden_nodes):
                train_net_layer_out[i] = tf.nn.relu(tf.matmul(train_net_layer_out[i - 1], weights[i]) + biases[i])
                valid_net_layer_out[i] = tf.nn.relu(tf.matmul(valid_net_layer_out[i - 1], weights[i]) + biases[i])
                test_net_layer_out[i] = tf.nn.relu(tf.matmul(test_net_layer_out[i - 1], weights[i]) + biases[i])
            else:
                train_net_layer_out[i] = tf.matmul(train_net_layer_out[i - 1], weights[i]) + biases[i]
                valid_net_layer_out[i] = tf.matmul(valid_net_layer_out[i - 1], weights[i]) + biases[i]
                test_net_layer_out[i] = tf.matmul(test_net_layer_out[i - 1], weights[i]) + biases[i]

    # Training computation.
    logits = train_net_layer_out[-1]
    with tf.name_scope('cross_entropy'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

    # Optimizer.
    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for the training, validation, and test data.
    with tf.name_scope("softmax"):
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(valid_net_layer_out[-1])
        test_prediction = tf.nn.softmax(test_net_layer_out[-1])

    with tf.name_scope('accuracy'):
        tf.summary.scalar('loss', loss)
        summary_op = tf.summary.merge_all()


num_steps = 10001

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

with tf.Session(graph=graph) as session:
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]

        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions, summary = session.run([optimizer, loss, train_prediction, summary_op], feed_dict=feed_dict)
        # _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        writer.add_summary(summary, step + i)
        if (step % 500 == 0):
            print("Minibatch loss at step %d: %f, %.1f%%, %.1f%%" % (step, l, accuracy(predictions, batch_labels), accuracy(valid_prediction.eval(), valid_labels)))
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
