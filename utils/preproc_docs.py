#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import os.path as osp
import pandas as pd
import numpy as np
import MeCab
import re
from bs4 import BeautifulSoup
import mojimoji
from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec
from multiprocessing import Pool


FN_STOP_WORDS = './jp_stop_words.txt'
NEOLOGD_PATH = '/usr/local/lib/mecab/dic/mecab-ipadic-neologd'
STOP_WORDS = [w for w in open(FN_STOP_WORDS).readlines() if w!='\n']


def normalize(word):
    word = word.lower()
    word = re.sub(r'\d', '0', word)
    return word


def clean_url(text):
    # text = BeautifulSoup(text, 'lxml').get_text()
    # <XXX>を削除してしまうが、XXXが重要な場合もあるのでコメントアウト
    text = re.sub(r'http\S+', '', text)
    return text


def clean_text(raw_text):
    text = clean_url(raw_text)
    text = mojimoji.han_to_zen(text)
    text = mojimoji.zen_to_han(text, kana=False)
    return text


def doc_to_words(raw_text):
    SELECTED_HINSIS = ['名詞', '動詞', '形容詞', '副詞']
    EXCLUDED_BUNRUI = ['非自立', '接尾']
    t = MeCab.Tagger('-d %s' % NEOLOGD_PATH)
    t.parse('')
    
    doc = clean_text(raw_text)
    words = []
    m = t.parseToNode(doc)
    while m:
        if ((m.feature.split(',')[0] in SELECTED_HINSIS) and
            (m.feature.split(',')[1] not in EXCLUDED_BUNRUI)):
            
            word = m.feature.split(',')[6]  # 原形
            if word == '*' and m.feature.split(',')[1] != '固有名詞':
                word = m.surface
            if word != '*':
                # preproc2
                word = normalize(word)
                if word not in STOP_WORDS:
                    words.append(word)
        m = m.next
    return words


def remove_ng_text(df):
    pat_ascii = re.compile(r'^[\x20-\x7E]+$')
    pat_css = re.compile(r"\';\n.+\(css\).appendTo\('head'\);")
    def is_all_ascii(content_str):
        result = pat_ascii.sub('', content_str)
        return len(result) == 0

    def ng_text(content_str):
        not_found_texts = ["404", "このページは300000文字以上のあるため、負荷対策のため表示できません。"]
        not_found = np.any([content_str.startswith(text) for text in not_found_texts])
        is_css = pat_css.match(content_str) is not None
        return not_found | is_css
                 
    mask_ng_title = df['title']=="contents api can not be analyzed"
    mask_ng_text = df['content'].apply(ng_text)
    mask_all_ascii = df['content'].apply(is_all_ascii)
    return df[~(mask_ng_title | mask_all_ascii | mask_ng_text)]
