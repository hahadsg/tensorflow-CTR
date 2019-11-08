# coding: utf-8
import os
import sys

import numpy as np
import tensorflow as tf

from common import (
    DScriteo,
    input_fn,
    make_model_fn,
    tf_debug_print,
)


def lr_arch_fn(features, labels, mode, params):
    num_fields = params['num_fields']
    num_features = params['num_features']
    l2_reg = params['l2_reg']

    feat_ids = tf.reshape(features['feat_ids'], shape=[-1, num_fields])  # None * n
    # feat_ids = tf_debug_print(feat_ids, "feat_ids")
    feat_vals = tf.reshape(features['feat_vals'], shape=[-1, num_fields])
    # feat_vals = tf_debug_print(feat_vals, "feat_vals")

    W = tf.get_variable(name='lr_w', shape=[num_features], initializer=tf.constant_initializer(0.0))
    # W = tf_debug_print(W, "W")
    b = tf.get_variable(name='lr_b', shape=[1], initializer=tf.constant_initializer(0.0))
    # b = tf_debug_print(b, "b")
    feat_w = tf.nn.embedding_lookup(W, feat_ids)  # None * n
    # feat_w = tf_debug_print(feat_w, "feat_w")
    logits = tf.reduce_sum(tf.multiply(feat_w, feat_vals), 1) + b  # None * 1
    # logits = tf_debug_print(logits, "logits")

    pred = tf.sigmoid(logits)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)) + \
           l2_reg * tf.nn.l2_loss(W)

    return loss, pred


def lr_default_params():
    params = dict({
        'ds_dir': '../data/criteo/train100k/',
        'model_dir': './model/lr',
        'num_epochs': 1,
        'batch_size': 32,
    })
    ds_obj = DScriteo(params['ds_dir'])
    params['ds_obj'] = ds_obj
    params['model_params'] = {
        'num_fields': 39,
        'num_features': ds_obj.num_features,
        'embedding_size': 8,
        'learning_rate': 0.01,
        'l2_reg': 0.0,
    }
    return params


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    lr_params = lr_default_params()

    ds_obj = lr_params['ds_obj']
    model_dir = lr_params['model_dir']
    num_epochs = 10
    batch_size = 32
    model_params = lr_params['model_params']
    print(model_params)

    model_fn = make_model_fn(lr_arch_fn)
    LR = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, params=model_params)

    train_spec = tf.estimator.TrainSpec(
        lambda: input_fn(ds_obj.file_tr, batch_size, num_epochs, True))
    eval_spec = tf.estimator.EvalSpec(
        lambda: input_fn(ds_obj.file_va, batch_size, 1),
        steps=None)
    tf.estimator.train_and_evaluate(LR, train_spec, eval_spec)

    """
    INFO:tensorflow:Saving dict for global step 28114: accuracy = 0.78262603, auc = 0.7328218, global_step = 28114, loss = 0.47694528
    INFO:tensorflow:Saving 'checkpoint_path' summary for global step 28114: ./model/lr\model.ckpt-28114
    INFO:tensorflow:Loss for final step: 0.24659579.
    """
