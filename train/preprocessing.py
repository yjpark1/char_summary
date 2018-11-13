import numpy as np
import pandas as pd
import glob
import re
from konlpy.tag import Komoran
from collections import Counter
from itertools import chain
import csv


def read_labeled_file(paths):
    # <read labeled text files>
    data = []
    for p in paths:
        file = pd.read_csv(p, encoding='cp949', engine='python')
        data.append(file)
    data = pd.concat(data)

    # <split text & summary>
    summary = data.body.values
    text = data.title.values + data.origin.values

    return summary, text


def remove_token(x):
    x = re.sub('\n+', ' ', x)
    x = re.sub('[!@#Δ\'\`·…]', '', x)
    x = re.sub('\s+', ' ', x)

    return x


def morph_num(x):
    x = komoran.pos(x)
    out = []
    for mrph, tg in x:
        if tg in ['SN']:
            mrph = '<num>'
        out.append(mrph)

    return out


def morph2doc(x, out_seq=False):
    x = ' '.join(x)
    if out_seq:
        x = '<start> ' + x + ' <stop>'
    return x


if __name__ == '__main__':
    paths = glob.glob('dataset/labeled_finance/*.csv')
    summary, text = read_labeled_file(paths)

    # remove token
    summary = [remove_token(x) for x in summary]
    text = [remove_token(x) for x in text]

    # morphology analysis
    komoran = Komoran()
    summary_morph = [morph_num(x) for x in summary]
    text_morph = [komoran.morphs(x) for x in text]

    # get number of tokens
    summary_morph_flat = list(chain.from_iterable(summary_morph))
    summary_cnt = Counter(summary_morph_flat)

    text_morph_flat = list(chain.from_iterable(text_morph))
    text_cnt = Counter(text_morph_flat)

    print('# tokens in summary: ', len(summary_cnt))
    print('# tokens in text: ', len(text_cnt))

    # save dictionary
    with open('token_text.csv', 'w') as f:  # Just use 'w' mode in 3.x
        w = csv.DictWriter(f, text_cnt.keys())
        w.writeheader()
        w.writerow(text_cnt)

    with open('token_summary.csv', 'w') as f:  # Just use 'w' mode in 3.x
        w = csv.DictWriter(f, summary_cnt.keys())
        w.writeheader()
        w.writerow(summary_cnt)

    # TODO: freq. 가 작은 token 제거

    # save text & summary for keras.prerocessing.text format

    text_morph = [morph2doc(x) for x in text_morph]
    summary_morph = [morph2doc(x, out_seq=True) for x in summary_morph]

    np.save('text.npy', text_morph)
    np.save('summary.npy', summary_morph)
