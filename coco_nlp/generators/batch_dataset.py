# encoding: utf-8

# author: cole.zhang
# contact: longzonejazz@gmail.com

from typing import Iterable, Iterator, TYPE_CHECKING
from typing import Any
from coco_nlp.generators import ABCGenerator
import math

if TYPE_CHECKING:
    from coco_nlp.processors.abc_processor import ABCProcessor

class BatchDataSet(Iterable):
    def __init__(self,
                 corpus: ABCGenerator,
                 *,
                 text_processor: 'ABCProcessor',
                 label_processor: 'ABCProcessor',
                 seq_length: int = None,
                 max_position: int = None,
                 segment: bool = False,
                 batch_size: int = 64) -> None:
        self.corpus = corpus
        self.text_processor = text_processor
        self.label_processor = label_processor

        self.seq_length = seq_length
        self.max_position = max_position
        self.segment = segment

        self.batch_size = batch_size

    def __len__(self) -> int:
        return math.floor(len(self.corpus) / self.batch_size)

    def __iter__(self) -> Iterator:
        batch_x, batch_y = [], []
        for x, y in self.corpus.sample():
            batch_x.append(x)
            batch_y.append(y)
            if len(batch_x) == self.batch_size:
                x_tensor = self.text_processor.transform(batch_x,
                                                         seq_length=self.seq_length,
                                                         max_position=self.max_position,
                                                         segment=self.segment)
                y_tensor = self.label_processor.transform(batch_y,
                                                          seq_length=self.seq_length,
                                                          max_position=self.max_position)
                yield x_tensor, y_tensor
                batch_x, batch_y = [], []

        if batch_x:
            x_tensor = self.text_processor.transform(batch_x,
                                                     seq_length=self.seq_length,
                                                     max_position=self.max_position,
                                                     segment=self.segment)
            y_tensor = self.label_processor.transform(batch_y,
                                                      seq_length=self.seq_length,
                                                      max_position=self.max_position)
            yield x_tensor, y_tensor

    def take(self, batch_count: int = None) -> Any:
        i = 0
        should_continue = True
        while should_continue:
            for batch_x, batch_y in self.__iter__():
                if batch_count is None or i < batch_count:
                    i += 1
                    yield batch_x, batch_y
                if batch_count and i >= batch_count:
                    should_continue = False
                    break
