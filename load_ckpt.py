import sys

import numpy as np
import time
from load_model.network import *
from load_model.layer import *
sys.path.append("../")




def dnn(input_shape=(None, 13), nb_classes=2):
    """
    The implementation of a DNN model
    :param input_shape: the shape of dataset
    :param nb_classes: the number of classes
    :return: a DNN model
    """
    activation = ReLU
    layers = [Linear(64),
              activation(),
              Linear(32),
              activation(),
              Linear(16),
              activation(),
              Linear(8),
              activation(),
              Linear(4),
              activation(),
              Linear(nb_classes),
              Softmax()]

    model = MLP(layers, input_shape)
    return model

def gradient_graph(x, preds, y=None):
    """
    Construct the TF graph of gradient
    :param x: the input placeholder
    :param preds: the model's symbolic output
    :return: the gradient graph
    """
    if y == None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = tf.reduce_max(preds, 1, keep_dims=True)
        y = tf.to_float(tf.equal(preds, preds_max))
        y = tf.stop_gradient(y)
    y = y / tf.reduce_sum(y, 1, keep_dims=True)

    # Compute loss
    loss = model_loss(y, preds, mean=False)

    # Define gradient of loss wrt input
    grad, = tf.gradients(loss, x)

    return grad

def model_loss(y, model, mean=True):
    """
    Define loss of TF graph
    :param y: correct labels
    :param model: output of the model
    :param mean: boolean indicating whether should return mean of loss
                 or vector of losses for each input of the batch
    :return: return mean of loss if True, otherwise return vector with per
             sample loss
    """

    op = model.op
    if op.type == "Softmax":
        logits, = op.inputs
    else:
        logits = model

    out = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)

    if mean:
        out = tf.reduce_mean(out)
    return out

def model_argmax(sess, x, predictions, samples, feed=None):
    """
    Helper function that computes the current class prediction
    :param sess: TF session
    :param x: the input placeholder
    :param predictions: the model's symbolic output
    :param samples: numpy array with input samples (dims must match x)
    :param feed: An optional dictionary that is appended to the feeding
             dictionary before the session runs. Can be used to feed
             the learning phase of a Keras model for instance
    :return: the argmax output of predictions, i.e. the current predicted class
    """
    feed_dict = {x: samples}
    if feed is not None:
        feed_dict.update(feed)
    probabilities = sess.run(predictions, feed_dict)

    if samples.shape[0] == 1:
        return np.argmax(probabilities)
    else:
        return np.argmax(probabilities, axis=1)


# prepare the testing data
# sample=[3,0,14,10,0,4,0,0,0,0,0,40,0]
sample1 = np.load('./data/data-x.npy')[1]
sample2 = np.load('./data/data-x.npy')[10]
# prepare the testing  model
input_shape = (None, 13)
nb_classes = 2
tf.set_random_seed(1234)
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.Session(config=config)
x = tf.placeholder(tf.float32, shape=input_shape)
y = tf.placeholder(tf.float32, shape=(None, nb_classes))
model = dnn(input_shape, nb_classes)
preds = model(x)
saver = tf.train.Saver()
model_path = 'model/' + 'test.model'
saver.restore(sess, model_path)

# construct the gradient graph
grad_0 = gradient_graph(x, preds)
# predict
if __name__ == '__main__':
    test_instances_array = np.load('./data/test_instances_set20220220134507.npy', allow_pickle=True)
    test_res_set = []
    test_accu = []
    for i in range(20):
        accuracy = 0.
        cor_cnt = 0
        wro_cnt = 0
        ground_truth_tmp = 0
        sample_id = 1
        for idx in test_instances_array[i]:
            test_res_tmp = []
            sample_tmp = np.load('./data/data-x.npy')[idx]
            label_tmp = model_argmax(sess, x, preds, np.array([sample_tmp]))
            ground_truth_tmp_array = np.load('./data/data-y.npy')[idx]
            if ground_truth_tmp_array[0] > 0:
                ground_truth_tmp = 0
            else:
                ground_truth_tmp = 1
            if label_tmp == ground_truth_tmp:
                test_res_tmp.append(1)
                cor_cnt += 1
                print("sample id: %d correct prediction, record 1 for this sample" % sample_id)
            else:
                test_res_tmp.append(0)
                wro_cnt += 1
                print("sample id: %d wrong prediction, record 0 for this sample" % sample_id)
            sample_id += 1
        test_res_set.append(test_res_tmp)
        accuracy = cor_cnt / (cor_cnt + wro_cnt)
        test_accu.append(accuracy)
        print("test id: %d total accuracy is %f" % (i+1, accuracy))
    time_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    array_to_save = np.array(test_accu, dtype=object)
    np.save('./test_accu' + time_str + '.npy', array_to_save)

    # label1 = model_argmax(sess, x, preds, np.array([sample1]))
    # label2 = model_argmax(sess, x, preds, np.array([sample2]))

