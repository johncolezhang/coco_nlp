# encoding: utf-8

# author: cole.zhang
# contact: longzonejazz@gmail.com

import unittest

import tests.test_classification.test_bi_lstm_model as base
from coco_nlp.embeddings import WordEmbedding
from coco_nlp.tasks.classification import CNN_GRU_Model
from tests.test_macros import TestMacros


class TestCNN_GRU_Model(base.TestBiLSTM_Model):
    @classmethod
    def setUpClass(cls):
        cls.EPOCH_COUNT = 1
        cls.TASK_MODEL_CLASS = CNN_GRU_Model
        cls.w2v_embedding = WordEmbedding(TestMacros.w2v_path)


if __name__ == "__main__":
    unittest.main()
