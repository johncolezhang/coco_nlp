# encoding: utf-8

# author: cole.zhang
# contact: longzonejazz@gmail.com

import unittest

from coco_nlp.generators import CorpusGenerator, BatchDataSet, MongoGenerator
from coco_nlp.processors import SequenceProcessor


class TestGenerator(unittest.TestCase):

    def test_batch_generator(self):
        text_processor = SequenceProcessor(
            segment=True
        )
        label_processor = SequenceProcessor(
            build_in_vocab='labeling',
            min_count=1,
            build_vocab_from_labels=True,
        )

        corpus_gen = MongoGenerator(
            db_name="spo",
            mongo_url="mongodb://localhost:27017",
            collection_name="test",
            buffer_size=3200
        )

        text_processor.build_vocab_generator([corpus_gen])
        label_processor.build_vocab_generator([corpus_gen])

        batch_dataset1 = BatchDataSet(corpus_gen,
                                      text_processor=text_processor,
                                      label_processor=label_processor,
                                      segment=False,
                                      seq_length=100,
                                      max_position=100,
                                      batch_size=12)

        print(len(batch_dataset1))

        duplicate_len = 100
        aaa = list(batch_dataset1.take(duplicate_len))
        assert len(aaa) == duplicate_len

        bbb = list(batch_dataset1.take(1))
        assert len(bbb) == 1

        ccc = list(batch_dataset1.take())
        print(len(ccc))

if __name__ == '__main__':
    unittest.main()
