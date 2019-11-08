# coding: utf-8
import os
import sys

import numpy as np
import tensorflow as tf


class DScriteo:
    def __init__(self, dir):
        self.dir = dir
        self.file_feature_map = os.path.join(self.dir, 'feature_map')
        self.file_tr = os.path.join(self.dir, 'tr.libsvm')
        self.file_va = os.path.join(self.dir, 'va.libsvm')

        self.num_fields = 39
        self.num_features = 0
        with open(self.file_feature_map, 'r') as f:
            for _ in f.readlines():
                self.num_features += 1


def tf_debug_print(var, name):
    info = "%s %s : " % (name, str(var.shape))
    return tf.Print(var, [var], info, 1, 100)


def input_fn(filenames, batch_size=32, num_epochs=1, is_shuffle=False):
    def decode_libsvm(line):
        columns = tf.string_split([line], ' ')
        label = tf.string_to_number(columns.values[0], tf.float32)
        splits = tf.string_split(columns.values[1:], ':')
        id_vals = tf.reshape(splits.values, splits.dense_shape)
        feat_ids, feat_vals = tf.split(id_vals, 2, 1)
        feat_ids = tf.string_to_number(feat_ids, tf.int32)
        feat_vals = tf.string_to_number(feat_vals, tf.float32)
        return {'feat_ids': feat_ids, 'feat_vals': feat_vals}, label

    ds = tf.data.TextLineDataset(filenames).map(decode_libsvm, 10).prefetch(100000)
    if is_shuffle:
        ds = ds.shuffle(buffer_size=batch_size*8)
    ds = ds.repeat(num_epochs).batch(batch_size)

    # return ds
    return ds.make_one_shot_iterator().get_next()


def make_model_fn(arch_fn):
    """
    need impl arch_fn(features, labels, mode, params) and return (loss, pred)
    """
    def model_fn(features, labels, mode, params):
        learning_rate = params['learning_rate']

        loss, pred = arch_fn(features, labels, mode, params)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        predictions = {'prob': pred}
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(labels, tf.math.greater_equal(pred, 0.5)),
            'auc': tf.metrics.auc(labels, pred),
        }

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops)

    return model_fn


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    # test input_fn
    ds_obj = DScriteo('../data/criteo/train100k/')
    input_fn = input_fn(ds_obj.file_tr, 2, 3)
    with tf.Session() as sess:
        for _ in range(2):
            print(sess.run(input_fn))

