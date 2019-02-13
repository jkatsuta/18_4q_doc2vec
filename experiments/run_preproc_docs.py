#!/usr/bin/env python
import os
import sys


preproc_mode = sys.argv[1]  # 'preproc1' / 'preproc2'

text_dir = './text'
out_dir = './preproc_data'
min_char = 300
max_char = 2000
if preproc_mode == 'preproc1':
    fn_words_list = 'words_list_mecab_pp1.pickle'
    fn_category_map = 'category_map_pp1.pickle'
elif preproc_mode == 'preproc2':
    fn_words_list = 'words_list_mecab_pp2.pickle'
    fn_category_map = 'category_map_pp2.pickle'

com = './preproc_docs.py '
com += '--text_dir %s ' % text_dir
com += '--out_dir %s ' % out_dir
com += '--min_char %d ' % min_char
com += '--max_char %d ' % max_char
com += '--fn_words_list %s ' % fn_words_list
com += '--fn_category_map %s ' % fn_category_map
com += '--preproc_mode %s ' % preproc_mode
print(com)
os.system(com)
