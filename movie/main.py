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
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CO
ECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import argparse
import os

import numpy as np
import tensorflow as tf

import nsml
from nsml import DATASET_PATH, HAS_DATASET, GPU_NUM, IS_ON_NSML
from dataset import MovieReviewDataset, preprocess

from keras import backend as K
from keras import models
from keras import layers
from keras import optimizers

# DONOTCHANGE: They are reserved for nsml
# This is for nsml leaderboard
def bind_model(model):
    # 학습한 모델을 저장하는 함수입니다.
    def save(filename, *args):
        filename = str(filename)
        model.save(filename)
    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(filename, *args):
        filename = str(filename)
        # identical to the previous one
        model.load_weights(filename)

    def infer(raw_data, **kwargs):
        """
        :param raw_data: raw input (여기서는 문자열)을 입력받습니다
        :param kwargs:
        :return:
        """
        preprocessed_data = preprocess(raw_data, 150)
        output_prediction = model.predict(preprocessed_data)[1].flatten().tolist()
        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(confidence interval, 포인트)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 confidence interval의 값은 영향을 미치지 않습니다
        return list(zip(np.zeros(len(output_prediction)), output_prediction))
    nsml.bind(save=save, load=load, infer=infer)


def _batch_loader(iterable, n=1):
    """
    데이터를 배치 사이즈만큼 잘라서 보내주는 함수입니다. Py
    의 DataLoader와 같은 역할을 합니다

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
    args.add_argument('--batch', type=int, default=512)
    args.add_argument('--strmaxlen', type=int, default=150)
    args.add_argument('--embedding', type=int, default=256)
    config = args.parse_args()

    inputs = layers.Input((config.strmaxlen,))
    layer = layers.Embedding(251, config.embedding, input_length=config.strmaxlen)(inputs)
    layer = layers.Bidirectional(layers.CuDNNGRU(512, return_sequences=True))(layer)
    layer = layers.Bidirectional(layers.CuDNNGRU(512, return_sequences=False))(layer)

    layer1 = layers.Dense(3)(layer)
    outputs1 = layers.Activation('softmax')(layer1)

    layer2 = layers.Dense(1)(layer1)
    outputs2 = layers.Activation('sigmoid')(layer2)
    outputs2 = layers.Lambda(lambda layer: layer * 9 + 1)(outputs2)
    model = models.Model(inputs=inputs, outputs=[outputs1, outputs2])
    model.summary()
    model.compile(optimizer=optimizers.Adam(lr=0.001, amsgrad=True, clipvalue=1.0), loss=['categorical_crossentropy', 'mse'], metrics=['accuracy'])
    
    # DONOTCHANGE: Reserved for nsml use
    bind_model(model)

    # DONOTCHANGE: They are reserved for nsml
    if config.pause:
        nsml.paused(scope=locals())


    # 학습 모드일 때 사용합니다. (기본값)
    if config.mode == 'train':
        # 데이터를 로드합니다.
        dataset = MovieReviewDataset(DATASET_PATH, config.strmaxlen)
        dataset_len = len(dataset)
        one_batch_size = dataset_len//config.batch
        if dataset_len % config.batch != 0:
            one_batch_size += 1
        # epoch마다 학습을 수행합니다.
        for epoch in range(config.epochs):
            avg_loss = 0.0
            avg_acc = 0.0
            dataset.shuffle()
            for i, (data, labels, sentiments) in enumerate(_batch_loader(dataset, config.batch)):
                loss, ce_loss, mse_loss, ce_acc, mse_acc = model.train_on_batch(data, [sentiments, labels])
                print('Batch : ', i + 1, '/', one_batch_size,
                      ', loss in this minibatch: ', float(loss),
                      ', CE in this minibatch: ', float(ce_loss),
                      ', CE ACC in this minibatch: ', float(ce_acc),
                      ', MSE in this minibatch: ', float(mse_loss))
                avg_loss += float(mse_loss)
                avg_acc += float(ce_acc)

            print('epoch:', epoch, ' train_loss:', float(avg_loss/one_batch_size), ' train_acc:', float(avg_acc/one_batch_size))
            nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=config.epochs,
                        train__loss=float(avg_loss/one_batch_size), step=epoch,
                        train_acc=float(avg_acc/one_batch_size))
            # DONOTCHANGE (You can decide how often you want to save the model)
            nsml.save(epoch)
