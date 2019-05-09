#!/usr/bin/python
# coding:utf-8
"""
@author: Cong Yu

@time: 2019/1/14 3:26 PM
"""

from collections import Counter
from preprocessing import CharacterIndexer, SlotIndexer, IntentIndexer
from gensim.models import Word2Vec
import json
import numpy as np
import pandas as pd
import pickle
import h5py
import math

from keras_contrib.layers import CRF
from keras_contrib.utils import save_load_utils
from keras_utilities import AttentionWithContext
from keras.models import model_from_json

sentindexer = pickle.load(open('encoded/atis_sentindexer.pkl', 'rb'))
slotindexer = pickle.load(open('encoded/atis_slotindexer.pkl', 'rb'))
intindexer  = pickle.load(open('encoded/atis_intindexer.pkl',  'rb'))

modelname = 'test_model'

import keras

with open('model/' + modelname + '.json') as f:
    json_string = f.read()

model = model_from_json(json_string, custom_objects={"CRF": CRF, "AttentionWithContext": AttentionWithContext})

model.load_weights('model/' + modelname + '.h5')

tst_text_idx = np.load('encoded/tst_text_idx.npy')
tst_char_idx = np.load('encoded/tst_char_idx.npy')
tst_slot_idx = np.load('encoded/tst_slot_idx.npy')
tst_int_idx  = np.load('encoded/tst_int_idx.npy')

prd_slots_dist, prd_ints_dist = model.predict([tst_text_idx, tst_char_idx])
prd_int_idx  = np.squeeze(np.argmax(prd_ints_dist, axis=-1))
prd_slot_idx = np.argmax(prd_slots_dist, axis=-1)
prd_ints = intindexer.inverse_transform(prd_int_idx)
tru_ints = intindexer.inverse_transform(tst_int_idx)

from sklearn.metrics import accuracy_score
print('INTENT ACC:  ', accuracy_score(tru_ints, prd_ints))


import re
def preprocess(snt):
    snt = snt.lower()
    snt = re.sub(r'[^0-9a-z\s]', '', snt)
    snt = snt.split()
    snt = ['BOS'] + snt + ['EOS']
    snt = [snt]
    out = sentindexer.transform(snt)
    return snt, out[0], out[1]


def predict(s):
    tk, wt, ct = preprocess(s)
    tk = tk[0]
    sp, ip = model.predict([wt, ct])
    sp = np.argmax(sp, axis=-1)
    ip = np.argmax(ip, axis=-1)
    sp = slotindexer.inverse_transform(np.expand_dims(sp, axis=-1))[0]
    sp = [x.split('-')[-1] for x in sp]

    spd = {}
    for i, p in enumerate(sp):
        if p != 'O':
            if p in spd.keys():
                spd[p].append(tk[i])
            else:
                spd[p] = []
                spd[p].append(tk[i])

    spo = {}
    for k in spd.keys():
        spo[k] = ' '.join(spd[k])

    ip = intindexer.inverse_transform([ip] + [[0]])[0]

    print('query:', s)
    print('slots:')
    print(spo)
    print('intent:', ip)

    return spo, ip


import time
t1 = time.time()
inpt = "looking for direct flights from Chicago to LAX"
a, b = predict(inpt)

t2 = time.time()
inpt = "give me flights and fares from New York to Dallas"
a, b = predict(inpt)

t3 = time.time()
inpt = "i want a "
a, b = predict(inpt)

t4 = time.time()


if __name__ == "__main__":
    print(t2-t1,t3-t2,t4-t3)