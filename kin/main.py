# -*- coding: utf-8 -*-

"""
Copyright 2018 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import argparse
import os

import numpy as np
import tensorflow as tf

import nsml
from nsml import DATASET_PATH, HAS_DATASET, IS_ON_NSML
from dataset import KinQueryDataset, preprocess

from keras import backend as K
from keras import models
from keras import layers
from keras import optimizers

# DONOTCHANGE: They are reserved for nsml
# This is for nsml leaderboard
def bind_model(model):
    # 학습한 모델을 저장하는 함수입니다.
    def save(file_name, *args):
        file_name = str(file_name)
        model.save(file_name)

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(file_name, *args):
        file_name = str(file_name)
        model.load_weights(file_name)

    def infer(raw_data, **kwargs):
        preprocessed_data = preprocess(raw_data, 150)
        pred = model.predict(preprocessed_data).flatten()
        clipped = np.where(pred > 0.5, 1, 0)
        return list(zip(pred.flatten().tolist(), clipped.flatten().tolist()))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)


def _batch_loader(iterable, n=1):
    """
    데이터를 배치 사이즈만큼 잘라서 보내주는 함수입니다. PyTorch의 DataLoader와 같은 역할을 합니다

    :param iterable: 데이터 list, 혹은 다른 포맷
    :param n: 배치 사이즈
    :return:
    """
    length = len(iterable)
    for n_idx in range(0, length, n):
        yield iterable[n_idx:min(n_idx + n, length)]


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # User options
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--epochs', type=int, default=100)
    args.add_argument('--batch', type=int, default=128)
    args.add_argument('--strmaxlen', type=int, default=150)
    args.add_argument('--embedding', type=int, default=256)
    args.add_argument('--threshold', type=float, default=0.5)
    config = args.parse_args()

    #------------------------------------------------------------------
    core_inputs = layers.Input((config.strmaxlen,))
    core_layer = layers.Embedding(252, config.embedding, input_length=config.strmaxlen)(core_inputs)
    core_layer = layers.Bidirectional(layers.CuDNNGRU(256, return_sequences=True))(core_layer)
    core_layer = layers.Bidirectional(layers.CuDNNGRU(256, return_sequences=False))(core_layer)
    core_model = models.Model(inputs=core_inputs, outputs=core_layer)

    inputs1 = layers.Input((config.strmaxlen,))
    inputs2 = layers.Input((config.strmaxlen,))

    layer1 = core_model(inputs1)
    layer2 = core_model(inputs2)
    main_layer = layers.Subtract()([layer1, layer2])
    main_layer = layers.Lambda(lambda layer: tf.norm(layer, ord=2, axis=-1, keep_dims=True))(main_layer)
    main_layer = layers.Dense(1)(main_layer)
    main_layer = layers.Activation('sigmoid')(main_layer)
    main_model = models.Model(inputs=[inputs1, inputs2], outputs=main_layer)
    main_model.summary()
    main_model.compile(optimizer=optimizers.Adam(lr=0.001, amsgrad=True), loss='binary_crossentropy', metrics=['accuracy'])
    #------------------------------------------------------------------

    # DONOTCHANGE: Reserved for nsml
    bind_model(main_model)

    # DONOTCHANGE: Reserved for nsml
    if config.pause:
        nsml.paused(scope=locals())

    if config.mode == 'train':
        # 데이터를 로드합니다.
        dataset = KinQueryDataset(DATASET_PATH, config.strmaxlen)
        dataset_len = len(dataset)
        one_batch_size = dataset_len//config.batch
        if dataset_len % config.batch != 0:
            one_batch_size += 1
        # epoch마다 학습을 수행합니다.
        for epoch in range(config.epochs):
            avg_loss = 0.0
            avg_acc = 0.0
            dataset.shuffle()
            for i, (data1, data2, labels) in enumerate(_batch_loader(dataset, config.batch)):
                loss, acc = main_model.train_on_batch([data1, data2], labels)
                print('Batch : ', i + 1, '/', one_batch_size,
                      ', BCE in this minibatch: ', float(loss),
                      ', ACC in this minibatch: ', float(acc))
                avg_loss += float(loss)
                avg_acc += float(acc)
            print('epoch:', epoch, ' train_loss:', float(avg_loss/one_batch_size), ' train_acc:', float(avg_acc/one_batch_size))
            nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=config.epochs,
                        train__loss=float(avg_loss/one_batch_size), step=epoch,
                        train_acc=float(avg_acc/one_batch_size))
            # DONOTCHANGE (You can decide how often you want to save the model)
            nsml.save(epoch)
