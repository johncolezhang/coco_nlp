# encoding: utf-8

# author: cole.zhang
# contact: longzonejazz@gmail.com


from typing import Dict, Any

from tensorflow import keras

from coco_nlp.layers import L
from coco_nlp.tasks.classification.abc_model import ABCClassificationModel


class BiGRU_Model(ABCClassificationModel):

    @classmethod
    def default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'layer_bi_gru': {
                'units': 128,
                'return_sequences': False
            },
            'layer_output': {
            }
        }

    def build_model_arc(self) -> None:
        output_dim = self.label_processor.vocab_size
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        layer_stack = [
            L.Bidirectional(L.GRU(**config['layer_bi_gru'])),
            L.Dense(output_dim, **config['layer_output']),
            self._activation_layer()
        ]

        tensor = embed_model.output
        for layer in layer_stack:
            tensor = layer(tensor)

        self.tf_model = keras.Model(embed_model.inputs, tensor)


if __name__ == "__main__":
    pass
