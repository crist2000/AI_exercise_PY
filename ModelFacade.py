from pinecone import Pinecone
from datasets import load_dataset
from pinecone.exceptions import PineconeException
import DataFrameHelper as DfHelper
import PineconeExt as pcEx
#import pandas as pd
import tools
from sentence_transformers import SentenceTransformer
import Constants as Const
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig
from transformers import pipeline
import os

class ModelFacade:

    def __init__(self, model_name):
        self.model_name = model_name
        self.model_path = f"{Const.models_dir}{model_name}\\"
        self.__model_HF = f"sentence-transformers/{self.model_name}"

    def save_model(self):
        if os.path.isdir(self.model_path):
            return
        else:
            os.mkdir(self.model_path)
            self.get_transformer().save_pretrained(self.model_path)

    def get_transformer(self):
        if os.path.isdir(self.model_path):
            return SentenceTransformer(self.model_path)
        else:
            return SentenceTransformer(self.__model_HF, token=Const.HF_TOKEN)

    def get_model_name(self):
        return self.model_name