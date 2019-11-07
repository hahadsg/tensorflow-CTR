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

        self.num_features = 0
        with open(self.file_feature_map, 'r') as f:
            for _ in f.readlines():
                self.num_features += 1


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
    if (is_shuffle):
        ds = ds.shuffle(buffer_size=256)
    ds = ds.repeat(num_epochs).batch(batch_size)

    # return ds
    return ds.make_one_shot_iterator().get_next()


def model_fn(features, labels, mode, params, config):
    # params
    num_fields = params['num_fields']
    num_features = params['num_features']
    l2_reg = params['l2_reg']
    learning_rate = params['learning_rate']

    print(features['feat_ids'].shape)
    feat_ids = tf.reshape(features['feat_ids'], shape=[-1, num_fields])  # None * n
    feat_vals = tf.reshape(features['feat_vals'], shape=[-1, num_fields])

    W = tf.get_variable(name='lr_w', shape=[num_features], initializer=tf.constant_initializer(0.0))
    b = tf.get_variable(name='lr_b', shape=[1], initializer=tf.constant_initializer(0.0))
    feat_w = tf.nn.embedding_lookup(W, feat_ids)  # None * n
    print(feat_w.shape)
    print(feat_vals.shape)
    print(tf.multiply(feat_w, feat_vals).shape)
    logits = tf.reduce_sum(tf.multiply(feat_w, feat_vals), 1) + b  # None * 1

    pred = tf.sigmoid(logits)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)) + \
        l2_reg * tf.nn.l2_loss(W)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    predictions = {'prob': pred}

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op)


if __name__ == '__main__':
    ds_obj = DScriteo('../data/criteo/train100k/')
    model_dir = './model/lr/'
    num_epochs = 3
    batch_size = 32

    params = {
        'num_fields': 39,
        'num_features': ds_obj.num_features,
        'embedding_size': 8,
        'learning_rate': 0.001,
        'l2_reg': 0.0,
    }
    print(params)

    LR = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, params=params)

    train_spec = tf.estimator.TrainSpec(lambda: input_fn(ds_obj.file_tr, batch_size, num_epochs))
    eval_spec = tf.estimator.EvalSpec(
        lambda: input_fn(ds_obj.file_tr, batch_size, 1),
        steps=None)
    tf.estimator.train_and_evaluate(LR, train_spec, eval_spec)

    # # test input_fn
    # input_fn = input_fn(ds_obj.file_tr, 2, 3)
    # with tf.Session() as sess:
    #     for _ in range(2):
    #         print(sess.run(input_fn))

