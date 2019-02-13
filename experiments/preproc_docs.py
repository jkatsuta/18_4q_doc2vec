#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import os.path as osp
import pickle
import argparse
from gensim.models.doc2vec import TaggedDocument
from preproc_module import doc_to_words1, doc_to_words2


def get_fn_texts(text_dir, each_num=None):
    SKIP_FILE = 'LICENSE.txt'
    fn_texts = []
    categories = []
    for p, dirs, files in os.walk(text_dir):
        if osp.basename(p) != 'text':
            cat_texts = [osp.join(p, f) for f in files if f != SKIP_FILE]
            if each_num is not None:
                cat_texts = cat_texts[:each_num]
            fn_texts += cat_texts
            categories += [osp.basename(p)] * len(cat_texts)
    return fn_texts, categories

        
def docs_to_words(fn_texts, min_char, max_char, preproc_mode):
    words_list = []
    for fn_text in fn_texts:
        if preproc_mode == 'preproc1':
            words = doc_to_words1(fn_text, min_char, max_char)
        elif preproc_mode == 'preproc2':
            words = doc_to_words2(fn_text, min_char, max_char)
        words_list.append(words)
    return words_list


def main(text_dir, out_dir, fn_words_list, fn_category_map,
         min_char=None, max_char=None, each_num=None, preproc_mode='preproc1'):
    fn_texts, categories = get_fn_texts(text_dir, each_num)
    words_list = docs_to_words(fn_texts, min_char, max_char, preproc_mode)

    tagged_words_list = []
    category_map = {}
    i = 0
    for words, category in zip(words_list, categories):
        if words is None:
            continue
        tag = 't%d' % i
        tagged_words_list.append(TaggedDocument(words=words, tags=[tag]))
        category_map[tag] = category
        i += 1

    if not osp.exists(out_dir):
        os.makedirs(out_dir)
    with open(osp.join(out_dir, fn_words_list), 'wb') as g1:
        pickle.dump(tagged_words_list, g1)
    with open(osp.join(out_dir, fn_category_map), 'wb') as g2:
        pickle.dump(category_map, g2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='preprocess version1')
    parser.add_argument('--text_dir', type=str, default='./text', help='directory of input text data')
    parser.add_argument('--out_dir', type=str, default='./preproc_data', help='directory of output preproc data')
    parser.add_argument('--min_char', type=int, default=None, help='doc is used only if char-len >= min_char')
    parser.add_argument('--max_char', type=int, default=None, help='char-len is truncated with upper limit of max_cahr')
    parser.add_argument('--each_num', type=int, default=None, help='number of doc for each category (test case)')
    parser.add_argument('--fn_words_list', type=str, default=None, help='file name of words-list output')
    parser.add_argument('--fn_category_map', type=str, default=None, help='file name of tag-category map output')
    parser.add_argument('--preproc_mode', type=str, default='preproc1', help='preprocess mode: preproc1/preproc2')
    args = parser.parse_args()
    args = vars(args)

    main(**args)
