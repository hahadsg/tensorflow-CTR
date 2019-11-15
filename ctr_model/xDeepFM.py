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


def xDeepFM_arch_fn(features, labels, mode, params):
    num_fields = params['num_fields']  # f
    num_features = params['num_features']  # n
    embedding_size = params['embedding_size']  # k
    deep_nodes = params['deep_nodes']  # DNN hidden layer node number
    CIN_nodes = params['CIN_nodes']
    is_batch_norm = params['is_batch_norm']
    l2_reg = params['l2_reg']

    feat_ids = tf.reshape(features['feat_ids'], shape=[-1, num_fields])  # None * f
    feat_vals = tf.reshape(features['feat_vals'], shape=[-1, num_fields])

    V = tf.get_variable(name='fm_v', shape=[num_features, embedding_size], initializer=tf.glorot_normal_initializer())
    W = tf.get_variable(name='fm_w', shape=[num_features], initializer=tf.constant_initializer(0.0))
    b = tf.get_variable(name='fm_b', shape=[1], initializer=tf.constant_initializer(0.0))
    CIN_params = []
    DNN_params = []

    with tf.variable_scope('Linear'):
        feat_w = tf.nn.embedding_lookup(W, feat_ids)  # None * f
        y_w = tf.reduce_sum(tf.multiply(feat_w, feat_vals), 1)  # None

    with tf.variable_scope('Embedding-Layer'):
        # 这里注意，xDeepFM原文是没有乘以x的
        # xDeepFM会将所有的特征都0/1化，所以不需要乘以x
        # 这里乘以x的原因是，我的特征数据中存在连续型变量，所以需要乘上x才会有差异化
        v = tf.nn.embedding_lookup(V, feat_ids)  # None * f * k
        x = tf.reshape(feat_vals, shape=[-1, num_fields, 1])  # None * f * 1
        vx = tf.multiply(v, x)  # None * f * k
        embeddings = tf.reshape(vx, shape=[-1, num_fields * embedding_size])  # None * (f*k)

    with tf.variable_scope('CIN'):
        CIN_out = []
        CIN_hidden_nums = []
        x_0 = tf.reshape(embeddings, shape=[-1, num_fields, embedding_size])  # None * f * k
        split_x_0 = tf.split(x_0, embedding_size, 2)  # k * None * f * 1
        CIN_hidden_nums.append(num_fields)
        x_l = x_0
        for i, num_node in enumerate(CIN_nodes):
            split_x_l = tf.split(x_l, embedding_size * [1], 2)  # k * None * H_{l-1} * 1
            # 左边shape: [k * None * f * 1]
            # 右边shape: [k * None * 1 * H_{l-1}] 因为transpose_b（相当于做了outer product）
            dot_result_m = tf.matmul(split_x_0, split_x_l, transpose_b=True)  # [k * None * f * H_{l-1}]
            dot_result_o = tf.reshape(dot_result_m, shape=[embedding_size, -1, num_fields * CIN_hidden_nums[-1]])
            dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])  # None * k * (f*H_{l-1})

            filters = tf.get_variable(name='CIN_f_%d' %i, shape=[1, CIN_hidden_nums[-1] * num_fields, num_node],
                                      initializer=tf.glorot_normal_initializer())
            CIN_params.append(filters)
            x_l = tf.nn.conv1d(dot_result, filters=filters, stride=1, padding='VALID')  # None * k * H_l
            x_l = tf.transpose(x_l, perm=[0, 2, 1])  # None * H_l * k

            CIN_out.append(x_l)
            CIN_hidden_nums.append(num_node)
        result = tf.concat(CIN_out, axis=1)  # None * (sum(H_i)) * k
        result = tf.reduce_sum(result, -1)  # None * (sum(H_i))

        w_o = tf.get_variable(name='CIN_w_output', shape=[np.sum(CIN_hidden_nums[1:])],
                              initializer=tf.glorot_normal_initializer())
        CIN_params.append(w_o)
        y_CIN = tf.reduce_sum(tf.multiply(result, w_o), 1)

    with tf.variable_scope('DNN'):
        h_l = embeddings
        for i, num_node in enumerate(deep_nodes):
            w_l = tf.get_variable(name='DNN_w_%d' % i, shape=[h_l.shape[1], num_node],
                                  initializer=tf.glorot_normal_initializer())
            b_l = tf.get_variable(name='DNN_b_%d' % i, shape=[1],
                                  initializer=tf.glorot_normal_initializer())
            DNN_params.append(w_l)
            h_l = tf.matmul(h_l, w_l) + b_l
            if is_batch_norm:
                h_l = tf.layers.batch_normalization(h_l, training=(mode == tf.estimator.ModeKeys.TRAIN))
            h_l = tf.nn.relu(h_l)
        w_o = tf.get_variable(name='DNN_w_output', shape=[deep_nodes[-1]],
                              initializer=tf.glorot_normal_initializer())
        DNN_params.append(w_o)
        y_DNN = tf.reduce_sum(tf.multiply(h_l, w_o), 1)

    logits = b + y_w + y_CIN + y_DNN  # None
    pred = tf.sigmoid(logits)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)) + \
        l2_reg * (tf.nn.l2_loss(W) + tf.nn.l2_loss(V))
    for w in DNN_params:
        loss += l2_reg * tf.nn.l2_loss(w)
    for w in CIN_params:
        loss += l2_reg * tf.nn.l2_loss(w)

    return loss, pred


def xDeepFM_default_params():
    ds_obj = DScriteo('../data/criteo/train100k/')
    params = dict({
        'ds_obj': ds_obj,
        'model_dir': './model/xDeepFM',
        'num_epochs': 1,
        'batch_size': 32,
    })
    params['model_params'] = {
        'num_fields': ds_obj.num_fields,
        'num_features': ds_obj.num_features,
        'embedding_size': 8,
        'deep_nodes': [100, 100],
        'CIN_nodes': [10, 11, 12],
        'is_batch_norm': True,
        'learning_rate': 0.001,
        'l2_reg': 1e-3,
    }
    return params


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    params = xDeepFM_default_params()
    config = estimator_default_config()

    ds_obj = params['ds_obj']
    model_dir = params['model_dir']
    num_epochs = 20
    batch_size = 32
    model_params = params['model_params']
    print(model_params)

    if os.path.exists(model_dir): shutil.rmtree(model_dir)  # clean model_dir
    model_fn = make_model_fn(xDeepFM_arch_fn)
    xDeepFM = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, params=model_params, config=config)

    xDeepFM.train(lambda: input_fn(ds_obj.file_tr, batch_size, num_epochs, True))
    print('eval in tr dataset')
    xDeepFM.evaluate(lambda: input_fn(ds_obj.file_tr, batch_size, 1))
    print('eval in va dataset')
    xDeepFM.evaluate(lambda: input_fn(ds_obj.file_va, batch_size, 1))

    """
    eval in tr dataset
    INFO:tensorflow:Saving dict for global step 56227: accuracy = 0.7805073, auc = 0.76424193, global_step = 56227, loss = 0.4829209
    eval in va dataset
    INFO:tensorflow:Saving dict for global step 56227: accuracy = 0.78272563, auc = 0.75082505, global_step = 56227, loss = 0.49588004
    """
