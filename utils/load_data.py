import pickle
import pandas as pd
import preproc_docs as ppdoc


def load_contents(fn_texts, mode='gmossp'):
    if mode == 'gmossp':
        load_gmossp_contents(fn_texts)


def load_gmossp_contents(fn_texts):
    dfs = [pd.read_json(fn_text, lines=True)
           for fn_text in fn_texts if osp.getsize(fn_text)]
    df = pd.concat(dfs)
    df = ppdoc.remove_ng_text(df)
    ma_ids = list(df['hash_id'].values)
    raw_texts = df['title'] + ' ' + df['content']

def preproc(fn_texts):
    dfs = [pd.read_json(fn_text, lines=True)
           for fn_text in fn_texts if osp.getsize(fn_text)]
    df = pd.concat(dfs)
    df = ppdoc.remove_ng_text(df)
    ma_ids = list(df['hash_id'].values)
    raw_texts = df['title'] + ' ' + df['content']

    words_list = []
    for raw_text in raw_texts.values:
        words = doc_to_words(raw_text)
        words_list.append(words)
    return ma_ids, words_list
    

def load_gmossp_contents(fn_texts, fn_ma_ids_list, fn_words_list):
    all_ma_ids, all_words_list = preproc(df)
    
    with open(fn_ma_ids_list, 'wb') as g1:
        pickle.dump(all_ma_ids, g1)
    with open(fn_words_list, 'wb') as g2:
        tagged_words_list =\
            [TaggedDocument(words=words, tags=[i])
             for i, words in enumerate(all_words_list)]
        pickle.dump(tagged_words_list, g2)