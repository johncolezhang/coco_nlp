from coco_nlp.generators import ABCGenerator
import numpy as np
from elasticsearch import Elasticsearch

class ESConnector:
    def __init__(
            self,
            es_host,
            es_port,
            es_username=None,
            es_password=None
    ):
        if isinstance(es_host, list):
            hosts = es_host
        elif isinstance(es_host, str):
            hosts = es_host.split(",")
        else:
            raise Exception("es host type should be list or str")

        self.es_conn = Elasticsearch(
            hosts=hosts,
            port=es_port,
            http_auth=(es_username, es_password)
        )

    def get_sample(self):
        return

    def get_random_sample(self):
        return

    def get_index_size(self):
        return


class ESGenerator:
    def __init__(
            self,
            es_host,
            es_port,
            index,
            buffer_size=2000,
            **kwargs
    ):
        self.es_conn = ESConnector(
            es_host=es_host,
            es_port=es_port,
            es_password=kwargs["es_password"] if "es_password" in kwargs.keys() else None,
            es_username=kwargs["es_username"] if "es_username" in kwargs.keys() else None,
        )

    def __iter__(self):
        # TODO: add iteration function
        yield "", ""

    def __len__(self):
        # TODO: add get size function
        return 0

    def sample(self):
        # TODO: add sampling function
        yield "", ""