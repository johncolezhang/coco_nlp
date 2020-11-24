# encoding: utf-8

# author: cole.zhang
# contact: longzonejazz@gmail.com


import os
import logging

logger = logging.Logger('coco_nlp', level='DEBUG')
stream_handler = logging.StreamHandler()

if os.environ.get('coco_nlp_DEV') == 'True':
    log_format = '%(asctime)s [%(levelname)s] %(name)s:%(filename)s:%(lineno)d - %(message)s'
else:
    log_format = '%(asctime)s [%(levelname)s] %(name)s - %(message)s'

stream_handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(stream_handler)

if __name__ == "__main__":
    pass
