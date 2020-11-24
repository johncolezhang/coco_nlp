# encoding: utf-8

# author: cole.zhang
# contact: longzonejazz@gmail.com

import os
import time
import unittest
import tempfile
from coco_nlp.tasks.seq2seq import Seq2Seq
from coco_nlp.corpus import ChineseDailyNerCorpus


class TestSeq2Seq(unittest.TestCase):
    def test_base_use_case(self):
        x, y = ChineseDailyNerCorpus.load_data('test')
        x = x[:200]
        y = y[:200]
        seq2seq = Seq2Seq(hidden_size=64,
                          encoder_seq_length=64,
                          decoder_seq_length=64)
        seq2seq.fit(x, y, epochs=1)
        res, att = seq2seq.predict(x)

        model_path = os.path.join(tempfile.gettempdir(), str(time.time()))
        seq2seq.save(model_path)

        s2 = Seq2Seq.load_model(model_path)
        res2, att2 = s2.predict(x)

        assert res2 == res
        assert (att2 == att).all()


if __name__ == '__main__':
    unittest.main()
