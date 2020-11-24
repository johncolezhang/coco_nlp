# encoding: utf-8

# author: cole.zhang
# contact: longzonejazz@gmail.com

from typing import Iterator
from typing import List, Any, Tuple
from coco_nlp.generators import ABCGenerator
import numpy as np

class CorpusGenerator(ABCGenerator):

    def __init__(self,
                 x_data: List,
                 y_data: List,
                 *,
                 buffer_size: int = 2000) -> None:
        super(CorpusGenerator, self).__init__(buffer_size=buffer_size)
        self.x_data = x_data
        self.y_data = y_data
        self.buffer_size = buffer_size

    def __iter__(self) -> Iterator[Tuple[Any, Any]]:
        for i in range(len(self.x_data)):
            yield self.x_data[i], self.y_data[i]

    def __len__(self) -> int:
        return len(self.x_data)

    def sample(self) -> Iterator[Tuple[Any, Any]]:
        buffer, is_full = [], False
        for sample in self:
            buffer.append(sample)
            if is_full:
                i = np.random.randint(len(buffer))
                yield buffer.pop(i)
            elif len(buffer) == self.buffer_size:
                is_full = True
        while buffer:
            i = np.random.randint(len(buffer))
            yield buffer.pop(i)