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


def DCN_arch_fn(features, labels, mode, params):
    num_fields = params['num_fields']  # f
    num_features = params['num_features']  # n
    embedding_size = params['embedding_size']  # k
    cross_layer_num = params['cross_network_layer_num']  # Cross Network layer number
    deep_nodes = params['deep_network_nodes']  # DNN hidden layer node number
    is_batch_norm = params['is_batch_norm']
    l2_reg = params['l2_reg']

    feat_ids = tf.reshape(features['feat_ids'], shape=[-1, num_fields])  # None * f
    feat_vals = tf.reshape(features['feat_vals'], shape=[-1, num_fields])

    V = tf.get_variable(name='feature_embedding', shape=[num_features, embedding_size], initializer=tf.glorot_normal_initializer())
    cross_W = []
    cross_b = []
    deep_W = []
    deep_b = []

    with tf.variable_scope('Embedding-Layer'):
        v = tf.nn.embedding_lookup(V, feat_ids)  # None * f * k
        x = tf.reshape(feat_vals, shape=[-1, num_fields, 1])  # None * f * 1
        vx = tf.multiply(v, x)  # None * f * k
        x_0 = tf.reshape(vx, shape=[-1, num_fields * embedding_size])  # None * (f*k)

    with tf.variable_scope('Cross-Network'):
        x_l = x_0
        for i in range(cross_layer_num):
            w_l = tf.get_variable(name='cross_w_%d' % i, shape=[num_fields * embedding_size, 1],
                                  initializer=tf.glorot_normal_initializer())
            b_l = tf.get_variable(name='cross_b_%d' % i, shape=[1],
                                  initializer=tf.glorot_normal_initializer())
            cross_W.append(w_l)
            cross_b.append(b_l)
            # calculate x' * w first
            xTw_l = tf.matmul(x_l, w_l)  # None * 1
            x_l = tf.multiply(x_0, xTw_l) + x_l + b_l

    with tf.variable_scope('Deep-Network'):
        h_l = x_0
        for i, num_node in enumerate(deep_nodes):
            w_l = tf.get_variable(name='deep_w_%d' % i, shape=[h_l.shape[1], num_node],
                                  initializer=tf.glorot_normal_initializer())
            b_l = tf.get_variable(name='deep_b_%d' % i, shape=[1],
                                  initializer=tf.glorot_normal_initializer())
            deep_W.append(w_l)
            deep_b.append(b_l)
            h_l = tf.matmul(h_l, w_l) + b_l
            if is_batch_norm:
                h_l = tf.layers.batch_normalization(h_l, training=(mode == tf.estimator.ModeKeys.TRAIN))
            h_l = tf.nn.relu(h_l)

    with tf.variable_scope('Combination_Layer'):
        x_stack = tf.concat([x_l, h_l], 1)
        logit_W = tf.get_variable(name='logit_w', shape=[x_stack.shape[1]],
                                  initializer=tf.glorot_normal_initializer())
        logit_b = tf.get_variable(name='logit_b', shape=[1],
                                  initializer=tf.glorot_normal_initializer())
        logits = tf.reduce_sum(tf.multiply(x_stack, logit_W), 1) + logit_b

    pred = tf.sigmoid(logits)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)) + \
           l2_reg * (tf.nn.l2_loss(logit_W) + tf.nn.l2_loss(V))
    for w in cross_W:
        loss += l2_reg * tf.nn.l2_loss(w)
    for w in deep_W:
        loss += l2_reg * tf.nn.l2_loss(w)

    return loss, pred


def DCN_default_params():
    ds_obj = DScriteo('../data/criteo/train100k/')
    params = dict({
        'ds_obj': ds_obj,
        'model_dir': './model/DCN',
        'num_epochs': 1,
        'batch_size': 32,
    })
    params['model_params'] = {
        'num_fields': ds_obj.num_fields,
        'num_features': ds_obj.num_features,
        'embedding_size': 8,
        'cross_network_layer_num': 4,
        'deep_network_nodes': [100, 100],
        'is_batch_norm': True,
        'learning_rate': 0.0001,
        'l2_reg': 1e-2,
    }
    return params


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    params = DCN_default_params()
    config = estimator_default_config()

    ds_obj = params['ds_obj']
    model_dir = params['model_dir']
    num_epochs = 20
    batch_size = 32
    model_params = params['model_params']
    print(model_params)

    if os.path.exists(model_dir): shutil.rmtree(model_dir)  # clean model_dir
    model_fn = make_model_fn(DCN_arch_fn)
    DCN = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, params=model_params, config=config)

    DCN.train(lambda: input_fn(ds_obj.file_tr, batch_size, num_epochs, True))
    print('eval in tr dataset')
    DCN.evaluate(lambda: input_fn(ds_obj.file_tr, batch_size, 1))
    print('eval in va dataset')
    DCN.evaluate(lambda: input_fn(ds_obj.file_va, batch_size, 1))

    """
    eval in tr dataset
    INFO:tensorflow:Saving dict for global step 56227: accuracy = 0.7748605, auc = 0.7500452, global_step = 56227, loss = 0.49158287
    eval in va dataset
    INFO:tensorflow:Saving dict for global step 56227: accuracy = 0.77625024, auc = 0.7450472, global_step = 56227, loss = 0.49128637
    """
