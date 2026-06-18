from pprint import pprint

from pinecone import Pinecone
from datasets import load_dataset
from pinecone.exceptions import PineconeException
import DataFrameHelper as DfHelper
import PineconeExt as pcEx
#import pandas as pd
import tools
import Constants as Const
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig
from transformers import pipeline
from tqdm.auto import tqdm
import ModelFacade as model

db_name="testdb1"
dn_name_ctx = "testdb-ctx"

model_id = "dslim/bert-base-NER"
tokenizer = AutoTokenizer.from_pretrained(Const.bert_path)#or model_id
config = AutoConfig.from_pretrained(Const.bert_path)#or model_id. not needed when connected to HF.
model_ner = AutoModelForTokenClassification.from_pretrained(Const.bert_path)#or model_id

##One time action to download this staff from HF to local drive for offline use.
#tokenizer.save_pretrained("C:\\Work")
#config.save_pretrained("C:\\Work")
#model_ner.save_pretrained("C:\\Work")

# #NLP pipeline
nlp = pipeline( "ner", model=model_ner, tokenizer=tokenizer, aggregation_strategy = "max", device="cpu")

try:
    df = DfHelper.pd.read_csv(Const.medium_articles_10)
    df_batch_size = 5

    mpnet_base_v2 = model.ModelFacade("all-mpnet-base-v2")
    retriever = mpnet_base_v2.get_transformer()

    df["values"], df["entities"] = DfHelper.get_entitiesANDembeddings(df["text_truncated"], retriever, nlp)
    df["metadata"] = df[["title","authors","tags","entities"]].to_dict("records")
    df["id"] = tools.make_ids(10)

    pc = Pinecone(api_key=Const.API_KEY)
    idx_ptr = pc.Index(name=db_name)
    pcEx.delete_all_records(idx_ptr)

    for i in (range(0, len(df), df_batch_size)):
        print(f"{i} {i+df_batch_size}")
        df_upsert = df[["id", "values", "metadata"]].iloc[i:i + df_batch_size]
        idx_ptr.upsert_from_dataframe(df_upsert)

except PineconeException as e:
    # Handle Pinecone-specific errors
    print(f"Pinecone error: {e}")
except Exception as e:
    # Handle other errors
    print(f"Unexpected error: {e}")




