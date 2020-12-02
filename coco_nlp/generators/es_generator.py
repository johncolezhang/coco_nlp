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

        if es_username and es_password:
            http_auth = (es_username, es_password)
        else:
            http_auth = None

        self.es = Elasticsearch(
            hosts=hosts,
            port=es_port,
            http_auth=http_auth
        )

    def get_sample(self, index, query_phrase):
        return

    def get_random_sample(self, index, size):
        return

    def get_index_size(self, index):
        resp = self.es.indices.stats(index=index)
        return resp["_all"]["primaries"]["docs"]["count"]


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


if __name__ == "__main__":
    es_conn = ESConnector(es_host="10.6.55.103", es_port=9200)
    print(es_conn.get_index_size(index="news"))