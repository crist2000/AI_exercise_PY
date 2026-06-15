from pprint import pprint

from pinecone import Pinecone
from pinecone.exceptions import PineconeException
import PineconeExt as pcEx
import pandas as pd
import tools
from sentence_transformers import SentenceTransformer
import Constants as Const

from tools import make_ids

dataset_file = "C:\\Work\\medium_post_titles.csv"
dataset_file_10 = "C:\\Work\\medium_post_titles_10.csv"
db_name="testdb1"
dn_name_ctx = "testdb-ctx"

try:
    df = pd.read_csv(dataset_file_10, nrows=10)
    # #print(df.isna().sum()) #isna() shows how many NULL values grouped by each column
    df.dropna(inplace=True)
    numrows = df["title"].count()
    #
    # #df selects False values from subtitle_truncated_flag column. ~ means NOT in pandas(literally  meaning> select which IS NOT True).
    # df = df[~df["subtitle_truncated_flag"]]
    # print(f"Number of unique categories:{df["category"].nunique()}")


    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', token=Const.HF_TOKEN )
    df["values"]=df["title"].map(lambda x: (model.encode(x)).tolist())
    df["id"] = make_ids(numrows)
    df["metadata"] = df.apply(lambda x:
        {
          "category": x["category"],
          "subtitle": x["subtitle"]
        }, axis=1)

    df_upsert=df[["id","values","metadata"]]

    pc = Pinecone(api_key=Const.API_KEY)
    idx_pointer = pc.Index(name=db_name)
    pcEx.delete_all_records(idx_pointer)

    idx_pointer.upsert_from_dataframe(df_upsert)


except PineconeException as e:
    # Handle Pinecone-specific errors
    print(f"Pinecone error: {e}")
except Exception as e:
    # Handle other errors
    print(f"Unexpected error: {e}")





