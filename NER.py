from pandas._libs.window import aggregations
from sympy.printing.numpy import const
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import torch
import PineconeExt as pcEx
import os
import Constants as Const
import pandas as pd
import tools as tools

def prepare_BERT_dataframe(model, df, numrows):
        df["values"]=df["title"].map(lambda x: (model.encode(x)).tolist())
        df["id"] = tools.make_ids(numrows)
        df["metadata"] = df.apply(lambda x:
            {
              "category": x["category"],
              "subtitle": x["subtitle"]
            }, axis=1)

        df_upsert=df[["id","values","metadata"]]
        return df_upsert


os.environ["HF_TOKEN"] = Const.HF_TOKEN

model_id = "dslim/bert-base-NER"
tokenzier = AutoTokenizer.from_pretrained(model_id)
ner_model = AutoModelForTokenClassification.from_pretrained(model_id)

nlp = pipeline( "ner",
                model=ner_model,
                tokenizer=tokenzier,
                device="cpu")

print(nlp("My name is Mark. I live in Sweden and work in Capgemini"))

#print(torch.__version__)
#print(torch.cuda.is_available())