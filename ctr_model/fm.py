# coding: utf-8
import os
import sys
import shutil

import numpy as np
import tensorflow as tf

from common import (
    DScriteo,
    input_fn,
    make_model_fn,
    estimator_default_config,
    tf_debug_print,
)


def fm_arch_fn(features, labels, mode, params):
    num_fields = params['num_fields']  # f
    num_features = params['num_features']  # n
    embedding_size = params['embedding_size']  # k
    l2_reg = params['l2_reg']

    feat_ids = tf.reshape(features['feat_ids'], shape=[-1, num_fields])  # None * f
    feat_vals = tf.reshape(features['feat_vals'], shape=[-1, num_fields])

    V = tf.get_variable(name='fm_v', shape=[num_features, embedding_size], initializer=tf.glorot_normal_initializer())
    W = tf.get_variable(name='lr_w', shape=[num_features], initializer=tf.constant_initializer(0.0))
    b = tf.get_variable(name='lr_b', shape=[1], initializer=tf.constant_initializer(0.0))

    with tf.variable_scope('1-way-term'):
        feat_w = tf.nn.embedding_lookup(W, feat_ids)  # None * f
        y_w = tf.reduce_sum(tf.multiply(feat_w, feat_vals), 1)  # None

    with tf.variable_scope('2-way-term'):
        v = tf.nn.embedding_lookup(V, feat_ids)  # None * f * k
        x = tf.reshape(feat_vals, shape=[-1, num_fields, 1])  # None * f * 1
        vx = tf.multiply(v, x)  # None * f * k
        square_sum = tf.square(tf.reduce_sum(vx, 1))  # None * k
        sum_square = tf.reduce_sum(tf.square(vx), 1)  # None * k
        y_v = 0.5 * tf.reduce_sum((square_sum - sum_square), 1)  # None

    logits = b + y_w + y_v  # None
    pred = tf.sigmoid(logits)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)) + \
           l2_reg * (tf.nn.l2_loss(W) + tf.nn.l2_loss(V))

    return loss, pred


def fm_default_params():
    ds_obj = DScriteo('../data/criteo/train100k/')
    params = dict({
        'ds_obj': ds_obj,
        'model_dir': './model/fm',
        'num_epochs': 1,
        'batch_size': 32,
    })
    params['model_params'] = {
        'num_fields': ds_obj.num_fields,
        'num_features': ds_obj.num_features,
        'embedding_size': 8,
        'learning_rate': 0.01,
        'l2_reg': 0.00005,
    }
    return params


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    params = fm_default_params()
    config = estimator_default_config()

    ds_obj = params['ds_obj']
    model_dir = params['model_dir']
    num_epochs = 20
    batch_size = 32
    model_params = params['model_params']
    print(model_params)

    if os.path.exists(model_dir): shutil.rmtree(model_dir)  # clean model_dir
    model_fn = make_model_fn(fm_arch_fn)
    FM = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, params=model_params, config=config)

    FM.train(lambda: input_fn(ds_obj.file_tr, batch_size, num_epochs, True))
    print('eval in tr dataset')
    FM.evaluate(lambda: input_fn(ds_obj.file_tr, batch_size, 1))
    print('eval in va dataset')
    FM.evaluate(lambda: input_fn(ds_obj.file_va, batch_size, 1))

    """
    eval in tr dataset
    INFO:tensorflow:Saving dict for global step 56227: accuracy = 0.7838643, auc = 0.732719, global_step = 56227, loss = 0.48970073
    eval in tr dataset
    INFO:tensorflow:Saving dict for global step 56227: accuracy = 0.78641164, auc = 0.73056126, global_step = 56227, loss = 0.48550275
    """
