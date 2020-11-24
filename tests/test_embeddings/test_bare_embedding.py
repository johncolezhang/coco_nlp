# encoding: utf-8

# author: cole.zhang
# contact: longzonejazz@gmail.com

import os
import time
import random
import tempfile
import unittest
from coco_nlp.logger import logger
from coco_nlp.processors import SequenceProcessor
from coco_nlp.corpus import SMP2018ECDTCorpus
from coco_nlp.embeddings import BareEmbedding
from coco_nlp.tasks.classification import BiGRU_Model
from coco_nlp.utils import load_data_object

sample_count = 50


class TestBareEmbedding(unittest.TestCase):

    def build_embedding(self):
        embedding = BareEmbedding()
        return embedding

    def test_base_cases(self):
        embedding = self.build_embedding()
        x, y = SMP2018ECDTCorpus.load_data()
        processor = SequenceProcessor()
        processor.build_vocab(x, y)
        embedding.setup_text_processor(processor)

        samples = random.sample(x, sample_count)
        res = embedding.embed(samples)
        max_len = max([len(i) for i in samples]) + 2

        if embedding.max_position is not None:
            max_len = embedding.max_position

        assert res.shape == (len(samples), max_len, embedding.embedding_size)

        # Test Save And Load
        embed_dict = embedding.to_dict()
        embedding2 = load_data_object(embed_dict)
        embedding2.setup_text_processor(processor)
        assert embedding2.embed(samples).shape == (len(samples), max_len, embedding.embedding_size)

    def test_with_model(self):
        x, y = SMP2018ECDTCorpus.load_data('test')
        embedding = self.build_embedding()

        model = BiGRU_Model(embedding=embedding)
        model.build_model(x, y)
        model_summary = []
        embedding.embed_model.summary(print_fn=lambda x: model_summary.append(x))
        logger.debug('\n'.join(model_summary))

        model.fit(x, y, epochs=1)

        model_path = os.path.join(tempfile.gettempdir(), str(time.time()))
        model.save(model_path)


if __name__ == "__main__":
    unittest.main()
