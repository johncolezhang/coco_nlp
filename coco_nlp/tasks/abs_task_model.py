# encoding: utf-8

# author: cole.zhang
# contact: longzonejazz@gmail.com


import json
import os
import pathlib
from abc import ABC, abstractmethod
from typing import Dict, Any, TYPE_CHECKING, Union

import tensorflow as tf

import coco_nlp
from coco_nlp.embeddings import ABCEmbedding
from coco_nlp.logger import logger
from coco_nlp.processors.abc_processor import ABCProcessor
from coco_nlp.utils import load_data_object
from coco_nlp.layers import KConditionalRandomField

if TYPE_CHECKING:
    from coco_nlp.tasks.labeling import ABCLabelingModel
    from coco_nlp.tasks.classification import ABCClassificationModel


class ABCTaskModel(ABC):

    def __init__(self) -> None:
        self.embedding: ABCEmbedding
        self.hyper_parameters: Dict[str, Any]
        self.sequence_length: int
        self.text_processor: ABCProcessor
        self.label_processor: ABCProcessor

        self.tf_model: tf.keras.Model

    def to_dict(self) -> Dict[str, Any]:
        model_json_str = self.tf_model.to_json()

        return {
            'tf_version': tf.__version__,  # type: ignore
            'coco_nlp_version': coco_nlp.__version__,
            '__class_name__': self.__class__.__name__,
            '__module__': self.__class__.__module__,
            'config': {
                'hyper_parameters': self.hyper_parameters,  # type: ignore
            },
            'embedding': self.embedding.to_dict(),  # type: ignore
            'text_processor': self.text_processor.to_dict(),
            'label_processor': self.label_processor.to_dict(),
            'tf_model': json.loads(model_json_str)
        }

    @classmethod
    def default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        """
        The default hyper parameters of the model dict, **all models must implement this function.**

        You could easily change model's hyper-parameters.

        For example, change the LSTM unit in BiLSTM_Model from 128 to 32.

            >>> from coco_nlp.tasks.classification import BiLSTM_Model
            >>> hyper = BiLSTM_Model.default_hyper_parameters()
            >>> print(hyper)
            {'layer_bi_lstm': {'units': 128, 'return_sequences': False}, 'layer_output': {}}
            >>> hyper['layer_bi_lstm']['units'] = 32
            >>> model = BiLSTM_Model(hyper_parameters=hyper)

        Returns:
            hyper params dict
        """
        raise NotImplementedError

    def save(self, model_path: str) -> str:
        """
        Save model
        Args:
            model_path:
        """
        pathlib.Path(model_path).mkdir(exist_ok=True, parents=True)
        model_path = os.path.abspath(model_path)

        with open(os.path.join(model_path, 'model_config.json'), 'w') as f:
            f.write(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
            f.close()

        self.embedding.embed_model.save_weights(os.path.join(model_path, 'embed_model_weights.h5'))
        self.tf_model.save_weights(os.path.join(model_path, 'model_weights.h5'))  # type: ignore
        logger.info('model saved to {}'.format(os.path.abspath(model_path)))
        return model_path

    @classmethod
    def load_model(cls, model_path: str) -> Union["ABCLabelingModel", "ABCClassificationModel"]:
        model_config_path = os.path.join(model_path, 'model_config.json')
        model_config = json.loads(open(model_config_path, 'r').read())
        model = load_data_object(model_config)

        model.embedding = load_data_object(model_config['embedding'])
        model.text_processor = load_data_object(model_config['text_processor'])
        model.label_processor = load_data_object(model_config['label_processor'])

        tf_model_str = json.dumps(model_config['tf_model'])

        print(tf_model_str)
        model.tf_model = tf.keras.models.model_from_json(tf_model_str,
                                                         custom_objects=coco_nlp.custom_objects)

        if isinstance(model.tf_model.layers[-1], KConditionalRandomField):
            model.layer_crf = model.tf_model.layers[-1]

        model.tf_model.load_weights(os.path.join(model_path, 'model_weights.h5'))
        model.embedding.embed_model.load_weights(os.path.join(model_path, 'embed_model_weights.h5'))
        return model

    @abstractmethod
    def build_model(self,
                    x_data: Any,
                    y_data: Any) -> None:
        raise NotImplementedError


if __name__ == "__main__":
    path = '/var/folders/x3/_dg9_drj42l_cc70tsqkpqrw0000gn/T/1590915853.4571211'
    m = ABCTaskModel.load_model(path)
    m.tf_model.summary()
