from pprint import pprint

from pinecone import Pinecone
from pinecone.exceptions import PineconeException
import PineconeExt as pcEx
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch

dataset_file = "C:\\Work\\medium_post_titles.csv"
db_name="testdb1"
dn_name_ctx = "testdb-ctx"

try:
    # df = pd.read_csv(dataset_file, nrows=10000)
    # #print(df["subtitle_truncated_flag"].value_counts())
    # print(df.shape)
    # #print(df.isna().sum()) #isna() shows how many NULL values grouped by each column
    # df.dropna(inplace=True)
    #
    # #df selects False values from subtitle_truncated_flag column. ~ means NOT in pandas(literally  meaning> select which IS NOT True).
    # df = df[~df["subtitle_truncated_flag"]]
    # print(df.shape)
    # print(f"Number of unique categories:{df["category"].nunique()}")

    sentences = ["This is an example sentence", "Each sentence is converted"]
    sentences1 = ["This is an example sentence"]

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', token=pcEx.HF_TOKEN )
    embeddings = model.encode(sentences1)
    print(len(embeddings[0]))

    #pc = Pinecone(api_key=pcEx.API_KEY)
    #idx_pointer = pc.Index(name=db_name)
    #pcEx.print_index(pc)



except PineconeException as e:
    # Handle Pinecone-specific errors
    print(f"Pinecone error: {e}")
except Exception as e:
    # Handle other errors
    print(f"Unexpected error: {e}")





