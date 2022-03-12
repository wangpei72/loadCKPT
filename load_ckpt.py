import sys

import numpy as np
import time
from load_model.network import *
from load_model.layer import *

sys.path.append("../")
from group_fairness_metric import statistical_parity_difference, equality_of_oppo
from group_fairness_metric import disparte_impact


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
    x_origin = np.load('./data/data-x.npy')
    y_origin = np.load('./data/data-y.npy')
    id_list = ['01', '02', '03', '04', '05']
    id_list_cnt = 0
    while id_list_cnt < 5:
        test_instances_array = np.load('data/test_instances_set' + id_list[id_list_cnt] + '.npy', allow_pickle=True)
        test_res_right_or_wrong_set = []  # 该list中的 0 1表示这次推理是错误或者正确的
        test_accu = []
        y_predict_20 = []  # 20组 y_predict shape[0]应该30
        eoop_20 = []
        eood_20 = []
        di_20 = []
        spd_20 = []
        print("id in id-list is %d " % id_list_cnt)
        for i in range(20):
            print('starting round %d' % i)
            accuracy = 0.
            cor_cnt = 0
            wro_cnt = 0
            ground_truth_tmp = 0
            eood = 0.
            eoop = 0.
            sample_id = 1
            y_pre = []
            X_sample = []
            y_true = []
            for idx in test_instances_array[i]:
                test_res_tmp = []
                sample_tmp = x_origin[idx]
                label_tmp = model_argmax(sess, x, preds, np.array([sample_tmp]))
                X_sample.append(sample_tmp)  # 保存当前的instance
                y_pre.append(label_tmp)  # 保存当前推理获得的y值 0 :<50k 1: >50k
                ground_truth_tmp_array = y_origin[idx]
                if ground_truth_tmp_array[0] > 0:
                    ground_truth_tmp = 0
                else:
                    ground_truth_tmp = 1
                y_true.append(ground_truth_tmp)
                if label_tmp == ground_truth_tmp:
                    test_res_tmp.append(1)
                    cor_cnt += 1
                    # print("sample id: %d correct prediction, record 1 for this sample" % sample_id)
                else:
                    test_res_tmp.append(0)
                    wro_cnt += 1
                    # print("sample id: %d wrong prediction, record 0 for this sample" % sample_id)
                sample_id += 1
            X_arr = np.array(X_sample, dtype=np.float32)
            y_arr = np.array(y_pre, dtype=np.float32)
            y_true_arr = np.array(y_true, dtype=np.float32)

            spd = statistical_parity_difference.S_P_D_adult_age(X_arr, y_arr)
            di = disparte_impact.D_I_adult_age(X_arr, y_arr)
            eoop = equality_of_oppo.E_Oppo_adult_age(X_arr, y_true_arr, y_arr)
            eood = equality_of_oppo.E_Odds_adult_age(X_arr, y_true_arr, y_arr)

            print("metrics: %f %f %f %f " % (spd, di, eoop, eood))
            y_predict_20.append(y_pre)
            eoop_20.append(eoop)
            eood_20.append(eood)
            di_20.append(di)
            spd_20.append(spd)

            test_res_right_or_wrong_set.append(test_res_tmp)
            accuracy = cor_cnt / (cor_cnt + wro_cnt)
            test_accu.append(accuracy)
            print("test id: %d total accuracy is %f" % (i + 1, accuracy))
        #     更改数据类型
        # accuracy_array = np.array(test_accu, dtype=np.float32)
        # predict_res_array = np.array(y_predict_20, dtype=np.float32)
        eoop_res_array = np.array(eoop_20, dtype=np.float32)
        eood_res_array = np.array(eood_20, dtype=np.float32)
        di_res_array = np.array(di_20, dtype=np.float32)
        spd_res_array = np.array(spd_20, dtype=np.float32)
        # np.save('./test_accuracy' + id_list[id_list_cnt] + '.npy', accuracy_array)
        # np.save('./adult-testres/y_predict' + id_list[id_list_cnt] + '.npy', predict_res_array)
        np.save('./adult-testres-age/eoop_res' + id_list[id_list_cnt] + '.npy', eoop_res_array)
        np.save('./adult-testres-age/eood_res' + id_list[id_list_cnt] + '.npy', eood_res_array)
        np.save('./adult-testres-age/di_res' + id_list[id_list_cnt] + '.npy', di_res_array)
        np.save('./adult-testres-age/spd_res' + id_list[id_list_cnt] + '.npy', spd_res_array)
        id_list_cnt += 1

