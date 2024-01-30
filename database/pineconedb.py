import itertools

from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm


class PineConeVDB:
    def __init__(self, api_key):
        self.api_key = api_key
        self.pc = Pinecone(api_key=api_key)

    def create_index(self, index_name, dimension=512, metric="cosine"):
        if index_name not in self.pc.list_indexes():
            self.pc.create_index(
                index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud="aws", region="us-west-2"),
            )
            print(f"Index {index_name} created successfully!")

        else:
            print("Index already exists!")

    def insert_records(self, index_name, vector_data):
        index = self.pc.Index(index_name)
        index.upsert(vectors=vector_data)
        print("Inserted successfully")

    def insert_batch(self, index_name, vectors_data):
        index = self.pc.Index(index_name)
        chunks_vectors_upsert = self.chunks(vectors_data, batch_size=100)
        for chunk_vectors in tqdm(
            chunks_vectors_upsert, desc="Inserting batch vectors"
        ):
            index.upsert(vectors=chunk_vectors)
        print("Inserted successfully")

    def delete(self, index_name, vector_id):
        index = self.pc.Index(index_name)
        index.delete(ids=vector_id)
        print("Deleted successfully!")

    @staticmethod
    def chunks(iterable, batch_size=100):
        """A helper function to break an iterable into chunks of size batch_size."""
        it = iter(iterable)
        chunk = tuple(itertools.islice(it, batch_size))
        while chunk:
            yield chunk
            chunk = tuple(itertools.islice(it, batch_size))
