import numpy as np
import pandas as pd
import glob
import re
import hgtk


def read_labeled_file(paths):
    # <read labeled text files>
    data = []
    for p in paths:
        file = pd.read_csv(p, encoding='cp949', engine='python')
        data.append(file)
    data = pd.concat(data)

    # <split text & summary>
    summary = data.body.values
    text = data.title.values + '. ' + data.origin.values

    return summary, text


def remove_token(x):
    x = re.sub('\n+', ' ', x)
    x = re.sub('[!@#Δ\'\`·…]', '', x)
    x = re.sub('\s+', ' ', x)

    return x


def morph2doc(x, out_seq=False):
    x = ' '.join(x)
    x = x.replace('  ', ' <s>')

    if out_seq:
        x = '<start> ' + x + ' <stop>'
    return x


if __name__ == '__main__':
    paths = glob.glob('datasets/labeled_finance/*.csv')
    summary, text = read_labeled_file(paths)

    # remove token
    summary = [remove_token(x) for x in summary]
    text = [remove_token(x) for x in text]

    # character-level decomposition
    text = [hgtk.text.decompose(x) for x in text]
    summary = [hgtk.text.decompose(x) for x in summary]

    # save text & summary for keras.prerocessing.text format
    text_char = [morph2doc(x) for x in text]
    summary_char = [morph2doc(x, out_seq=True) for x in summary]

    np.save('datasets/text.npy', text_char)
    np.save('datasets/summary.npy', summary_char)
