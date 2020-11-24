# encoding: utf-8

# author: cole.zhang
# contact: longzonejazz@gmail.com

import unittest

from tensorflow.keras.utils import get_file

from coco_nlp.embeddings import WordEmbedding
from coco_nlp.macros import DATA_PATH
from tests.test_embeddings.test_bare_embedding import TestBareEmbedding


class TestWordEmbedding(TestBareEmbedding):

    def build_embedding(self):
        sample_w2v_path = get_file('sample_w2v.txt',
                                   "http://s3.bmio.net/coco_nlp/sample_w2v.txt",
                                   cache_dir=DATA_PATH)
        embedding = WordEmbedding(sample_w2v_path)
        return embedding


if __name__ == '__main__':
    unittest.main()
