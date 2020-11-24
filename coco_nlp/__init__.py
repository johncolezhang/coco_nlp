import os
from typing import Dict, Any

os.environ['TF_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

custom_objects: Dict[str, Any] = {}

from coco_nlp import layers
from coco_nlp.utils.dependency_check import dependency_check

custom_objects = layers.resigter_custom_layers(custom_objects)
dependency_check()
