#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : cole.zhang
# Site    : https://eliyar.biz

# Time    : 2020/9/1 10:26 下午
# File    : bi_gru_crf_model.py
# Project : coco_nlp

from typing import Dict, Any

from tensorflow import keras

from coco_nlp.layers import L, KConditionalRandomField
from coco_nlp.tasks.labeling.abc_model import ABCLabelingModel


class BiGRU_CRF_Model(ABCLabelingModel):

    @classmethod
    def default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'layer_bgru': {
                'units': 128,
                'return_sequences': True
            },
            'layer_dropout': {
                'rate': 0.4
            },
            'layer_time_distributed': {},
            'layer_activation': {
                'activation': 'softmax'
            }
        }

    def build_model_arc(self) -> None:
        output_dim = self.label_processor.vocab_size

        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        crf = KConditionalRandomField()

        layer_stack = [
            L.Bidirectional(L.GRU(**config['layer_bgru']), name='layer_bgru'),
            L.Dropout(**config['layer_dropout'], name='layer_dropout'),
            L.Dense(output_dim, **config['layer_time_distributed']),
            crf
        ]

        tensor = embed_model.output
        for layer in layer_stack:
            tensor = layer(tensor)

        self.tf_model = keras.Model(embed_model.inputs, tensor)
        self.crf_layer = crf

    def compile_model(self,
                      loss: Any = None,
                      optimizer: Any = None,
                      metrics: Any = None,
                      **kwargs: Any) -> None:
        if loss is None:
            loss = self.crf_layer.loss
        if metrics is None:
            metrics = [self.crf_layer.accuracy]
        super(BiGRU_CRF_Model, self).compile_model(loss=loss,
                                                   optimizer=optimizer,
                                                   metrics=metrics,
                                                   **kwargs)


if __name__ == "__main__":
    from coco_nlp.corpus import ChineseDailyNerCorpus
    from coco_nlp.callbacks import EvalCallBack

    train_x, train_y = ChineseDailyNerCorpus.load_data('train')
    valid_x, valid_y = ChineseDailyNerCorpus.load_data('valid')
    test_x, test_y = ChineseDailyNerCorpus.load_data('test')

    model = BiGRU_CRF_Model(sequence_length=10)

    eval_callback = EvalCallBack(coco_model=model,
                                 x_data=valid_x,
                                 y_data=valid_y,
                                 truncating=True,
                                 step=1)

    model.fit(train_x, train_y, valid_x, valid_y, epochs=1,
              callbacks=[])
    y = model.predict(test_x[:200])
    model.tf_model.summary()
    model.evaluate(test_x, test_y)
