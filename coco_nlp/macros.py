# encoding: utf-8

# author: cole.zhang
# contact: longzonejazz@gmail.com


import os
from pathlib import Path
from typing import Dict

DATA_PATH = os.path.join(str(Path.home()), '.coco_nlp')

Path(DATA_PATH).mkdir(exist_ok=True, parents=True)


class Config:

    def __init__(self) -> None:
        self.verbose = False

    def to_dict(self) -> Dict:
        return {
            'verbose': self.verbose
        }


config = Config()

if __name__ == "__main__":
    pass
