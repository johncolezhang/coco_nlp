# encoding: utf-8

# author: cole.zhang
# contact: longzonejazz@gmail.com

from typing import Dict, Any

from tensorflow import keras

from coco_nlp.layers import L
from coco_nlp.tasks.classification.abc_model import ABCClassificationModel


class CNN_LSTM_Model(ABCClassificationModel):
    @classmethod
    def default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'conv1d_layer': {
                'filters': 128,
                'kernel_size': 5,
                'activation': 'relu'
            },
            'max_pool_layer': {},
            'lstm_layer': {
                'units': 100
            },
            'layer_output': {

            },
        }

    def build_model_arc(self) -> None:
        output_dim = self.label_processor.vocab_size

        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        # build model structure in sequent way
        layer_stack = [
            L.Conv1D(**config['conv1d_layer']),
            L.MaxPooling1D(**config['max_pool_layer']),
            L.LSTM(**config['lstm_layer']),
            L.Dense(output_dim, **config['layer_output']),
            self._activation_layer()
        ]

        tensor = embed_model.output
        for layer in layer_stack:
            tensor = layer(tensor)

        self.tf_model = keras.Model(embed_model.inputs, tensor)


if __name__ == "__main__":
    pass
