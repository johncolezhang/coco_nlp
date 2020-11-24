# encoding: utf-8

# author: cole.zhang
# contact: longzonejazz@gmail.com

from tensorflow.keras.utils import get_file

from coco_nlp.embeddings import BertEmbedding
from coco_nlp.macros import DATA_PATH
from tests.test_embeddings.test_bare_embedding import TestBareEmbedding


class TestTransferEmbedding(TestBareEmbedding):

    def build_embedding(self):
        bert_path = get_file('bert_sample_model',
                             "http://s3.bmio.net/coco_nlp/bert_sample_model.tar.bz2",
                             cache_dir=DATA_PATH,
                             untar=True)
        embedding = BertEmbedding(model_folder=bert_path)
        return embedding


if __name__ == "__main__":
    pass
