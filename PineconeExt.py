from pinecone import ServerlessSpec
from pinecone.exceptions import PineconeException

API_KEY = ""
HF_TOKEN = ""
ENV = "us-east-1"

def create_index(pclient, name, dim, metric = "cosine"):
    pclient.create_index(name=name, dimension=dim,spec=ServerlessSpec(region=ENV, cloud="aws"), metric=metric)

def print_index(pclient):
    for item in pclient.list_indexes().names():
        print(item)

def upsert_index(idx_pointer, data):
    #data_example = [
    #    {
    #"id": "A",
    #"values": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    #"metadata": {"genre": "comedy", "year": 2020}
    #}]

    idx_pointer.upsert(vectors=data)

def update_index(idx_pointer, id:str, values:list, metadata:dict):
    idx_pointer.update(
        #namespace="example-namespace",
        id=id,
        values=values,
        set_metadata=metadata)

def query_index(idx_pointer, id:str, values:list, metadata:dict):
    res = idx_pointer.query(
        # namespace="example-namespace",
        id=id,
        vector=values,
        filter=metadata,
        # filter={
        #    "genre": {"$eq": "SciFi"}
        # },
        top_k=3,
        include_values=True
    )

    for item in res.matches:
        print(f"{item.id}::{item.score}::{item.values}")

def fetch_index_byId(idx_pointer, str_id):
    res = idx_pointer.fetch(
        ids=[str_id]
        #namespace="example-namespace"
        )

    dic = res.vectors[str_id].to_dict()
    #print the same outcome
    print(dic["values"])
    print(res.vectors[str_id].values)

def print_namespaces(idx_pointer):
    for page in idx_pointer.list_namespaces():
        for ns in page.namespaces:
            print(ns.name, ns.record_count)
