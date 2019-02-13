import MeCab
import re
from bs4 import BeautifulSoup
import mojimoji


NEOLOGD_PATH = '/usr/local/lib/mecab/dic/mecab-ipadic-neologd'
FN_STOP_WORDS = './jp_stop_words.txt'
STOP_WORDS = [w for w in open(FN_STOP_WORDS).readlines() if w!='\n']


def doc_to_words1(fn_text, min_char, max_char):
    t = MeCab.Tagger('-d %s' % NEOLOGD_PATH)
    t.parse('')
    
    print(fn_text)
    doc = preproc1(fn_text, min_char, max_char)
    if doc is None:
        return None
    
    words = []
    m = t.parseToNode(doc)
    while m:
        word = m.surface
        if len(word) > 0:
            words.append(word)
        m = m.next
    return words


def preproc1(fn_text, min_char=None, max_char=None):
    raw_text = open(fn_text).read()
    text = _truncate(raw_text, min_char, max_char)
    return text


def _truncate(raw_text, min_char=None, max_char=None):
    def check_min_char(text, min_char):
        if min_char is not None and len(text) < min_char:
            return None
        return text
    
    # "[2:]"にしてるのは、livedoor-newsの最初の2行（URL, 時間）を削除するため。
    truncate_texts = raw_text.split('\n')[2:]
    # 関連リンクは後ろから探す。
    for i, line in enumerate(truncate_texts[::-1]):
        if line.find('関連リンク') >= 0 or line.find('関連記事') >= 0:
            truncate_texts = truncate_texts[:-(i+1)]
            break
    
    if max_char is None:
        text = ' '.join(truncate_texts)
        return check_min_char(text, min_char)
    
    text = ''
    for para in truncate_texts:
        if len(text) > max_char:
            break
        text += ' ' + para
    return check_min_char(text, min_char)

# -----

def doc_to_words2(fn_text, min_char, max_char):
    SELECTED_HINSIS = ['名詞', '動詞', '形容詞', '副詞']
    EXCLUDED_BUNRUI = ['非自立', '接尾']
    t = MeCab.Tagger('-d %s' % NEOLOGD_PATH)
    t.parse('')

    print(fn_text)
    doc = preproc2(fn_text, min_char, max_char)
    if doc is None:
        return None

    words = []
    m = t.parseToNode(doc)
    while m:
        if (m.surface != '\u3000' and
            (m.feature.split(',')[0] in SELECTED_HINSIS) and
            (m.feature.split(',')[1] not in EXCLUDED_BUNRUI)):

            word = m.feature.split(',')[6]  # 原形
            if word == '*' and m.feature.split(',')[1] != '固有名詞':
                word = m.surface
            if word != '*':
                # additional preproc
                word = _normalize(word)
                if word not in STOP_WORDS:
                    words.append(word)
        m = m.next
    return words


def preproc2(fn_text, min_char=None, max_char=None):
    raw_text = open(fn_text).read()
    text = _truncate(raw_text, min_char, max_char)
    if text is not None:
        text = _clean_url(text)
        text = mojimoji.han_to_zen(text)
        text = mojimoji.zen_to_han(text, kana=False)
    return text


def _clean_url(text):
    text = BeautifulSoup(text, 'lxml').get_text()
    text = re.sub(r'http\S+', '', text)
    return text


def _normalize(word):
    word = word.lower()
    word = re.sub(r'\d', '0', word)
    return word
