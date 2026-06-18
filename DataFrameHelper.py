import pandas as pd
from tqdm.auto import tqdm


# #df selects False values from subtitle_truncated_flag column. ~ means NOT in pandas(literally  meaning> select which IS NOT True).
# df = df[~df["subtitle_truncated_flag"]]
# print(f"Number of unique categories:{df["category"].nunique()}")

def get_named_entities(df_text_list, pipeline):
    entities_list = []

    for doc in df_text_list:
        ## to print all words to 1 row
        print([f"{item["word"]}-{item["entity_group"]}-{item["score"]}" for item in (pipeline(doc))])#to print in 1 row
        entities_list.append([item["word"] for item in (pipeline(doc))])

    return entities_list

def get_entitiesANDembeddings(df_text_list, transformer, pipeline):
    emb_list = []
    entities_list = []

    for doc in tqdm(df_text_list):
        emb_list.append(transformer.encode(doc).tolist())
        entities_list.append([item["word"] for item in (pipeline(doc))])

    return emb_list, entities_list

def get_embeddings(df_text_list, transformer):
    emb_list = df_text_list.map(lambda x: (transformer.encode(x)).tolist())
    return emb_list

def clean_NA(data_fr):
    print(data_fr.isna().sum()) #isna() shows how many NULL values grouped by each column
    data_fr.dropna(inplace=True) #remove data records from DataFrame where at least one column contains NULL value

    return data_fr