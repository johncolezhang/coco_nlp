from pymongo import MongoClient
from coco_nlp.generators import ABCGenerator
import numpy as np

class MongoConnector:
    def __init__(self, mongo_url, db_name, collection_name):
        self.client = MongoClient(mongo_url)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def insert_record(self, record_list):
        resp = self.collection.insert_many(record_list)
        return resp.inserted_ids

    def get_collection_count(self):
        return self.collection.count_documents({})

    def get_random_sample(self, batch_size=20, **kwargs):
        """
        return generator should be like
        [
        {"_id": xxx, "x": ["a", "b", "c"], "y": ["O", "O", "O"]},
        {"_id": xxx, "x": ["q", "w"], "y": ["B-SUBJ", "I-SUBJ"]},
        ...
        ]
        """
        if 'sample_size' in kwargs:
            return self.collection.aggregate([{"$sample": {"size": kwargs['sample_size']}}])
        else:
            return self.collection.find({}).batch_size(batch_size)


class MongoGenerator(ABCGenerator):
    def __init__(
            self,
            db_name,
            collection_name,
            mongo_url,
            buffer_size: int = 2000,
    ):
        self.buffer_size = buffer_size
        self.mongo_conn = MongoConnector(
            mongo_url=mongo_url,
            db_name=db_name,
            collection_name=collection_name
        )

    def __iter__(self):
        all_data = self.mongo_conn.get_random_sample()
        for ad in all_data:
            yield ad["x"], ad["y"]

    def __len__(self):
        return self.mongo_conn.get_collection_count()

    def sample(self):
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
