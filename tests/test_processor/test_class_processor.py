# encoding: utf-8

# author: cole.zhang
# contact: longzonejazz@gmail.com

import unittest
from tests.test_macros import TestMacros

from coco_nlp.utils import load_data_object
from coco_nlp.generators import CorpusGenerator
from coco_nlp.processors import ClassificationProcessor


class TestClassificationProcessor(unittest.TestCase):
    def test_processor(self):
        x_set, y_set = TestMacros.load_classification_corpus()
        processor = ClassificationProcessor()
        processor.build_vocab(x_set, y_set)
        transformed_idx = processor.transform(y_set[20:40])

        info_dict = processor.to_dict()

        p2: ClassificationProcessor = load_data_object(info_dict)
        assert (transformed_idx == p2.transform(y_set[20:40])).all()
        assert y_set[20:40] == p2.inverse_transform(transformed_idx)

    def test_multi_label_processor(self):
        from coco_nlp.corpus import JigsawToxicCommentCorpus
        file_path = TestMacros.jigsaw_mini_corpus_path
        corpus = JigsawToxicCommentCorpus(file_path)
        x_set, y_set = corpus.load_data()

        corpus_gen = CorpusGenerator(x_set, y_set)

        processor = ClassificationProcessor(multi_label=True)
        processor.build_vocab_generator([corpus_gen])
        transformed_idx = processor.transform(y_set[20:40])

        info_dict = processor.to_dict()

        p2: ClassificationProcessor = load_data_object(info_dict)
        assert (transformed_idx == p2.transform(y_set[20:40])).all()

        x1s = y_set[20:40]
        x2s = p2.inverse_transform(transformed_idx)
        for sample_x1, sample_x2 in zip(x1s, x2s):
            assert sorted(sample_x1) == sorted(sample_x2)


if __name__ == "__main__":
    pass
