# encoding: utf-8

# author: cole.zhang
# contact: longzonejazz@gmail.com

from abc import ABC
from typing import Iterable, Iterator
from typing import Any, Tuple

import numpy as np

class ABCGenerator(Iterable, ABC):
    def __init__(self, buffer_size: int = 2000) -> None:
        self.buffer_size = buffer_size

    def __iter__(self) -> Iterator[Tuple[Any, Any]]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def sample(self) -> Iterator[Tuple[Any, Any]]:
        raise NotImplementedError