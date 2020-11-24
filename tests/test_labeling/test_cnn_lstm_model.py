# encoding: utf-8

# author: cole.zhang
# contact: longzonejazz@gmail.com

import unittest

import tests.test_labeling.test_bi_lstm_model as base
from coco_nlp.tasks.labeling import CNN_LSTM_Model


class TestCNN_LSTM_Model(base.TestBiLSTM_Model):

    @classmethod
    def setUpClass(cls):
        cls.EPOCH_COUNT = 1
        cls.TASK_MODEL_CLASS = CNN_LSTM_Model


if __name__ == "__main__":
    unittest.main()
