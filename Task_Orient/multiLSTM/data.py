#!/usr/bin/python
# coding:utf-8
"""
@author: Cong Yu

@time: 2019/1/10 1:53 PM
"""
from collections import Counter
from preprocessing import CharacterIndexer, SlotIndexer, IntentIndexer
from gensim.models import Word2Vec
import json
import numpy as np
import pandas as pd
import pickle

def readatis(filename='data/atis/atis-2.train.w-intent.iob'):
    """
    function for reading the ATIS
    """
    data = pd.read_csv(filename, sep='\t', header=None)
    # get sentences and ner labels
    sents = [s.split() for s in data[0].tolist()]
    ners  = [s.split() for s in data[1].tolist()]
    # for sents, replace digits
    for i, sent in enumerate(sents):
        sent = ' '.join(sent)
        for d in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            sent = sent.replace(d, '#')
        sents[i] = sent.split()
    # check lengths
    assert(len(sents)==len(ners))
    # the intent label is the last item of ners.
    # remove it and replace it with a 'O' null tag
    ints = [s[-1] for s in ners]
    ners = [s[:-1]+['O'] for s in ners]
    # check sent, ner, int lengths
    assert(len(sents)==len(ints))
    for i in range(len(sents)):
        assert(len(sents[i])==len(ners[i]))
    return sents, ners, ints

if __name__ == "__main__":
    trn_texts, trn_slots, trn_ints = readatis('./data/atis/atis-2.train.w-intent.iob')
    dev_texts, dev_slots, dev_ints = readatis('./data/atis/atis-2.dev.w-intent.iob')
    tst_texts, tst_slots, tst_ints = readatis('./data/atis/atis.test.w-intent.iob')
    print(len(trn_texts), len(dev_texts), len(tst_texts))
    slots_len = len(list(set([t for s in trn_slots for t in s])))
    intent_len = len(list(set(trn_ints)))
    slots_c = Counter([t for s in trn_slots for t in s]).most_common(10)
    intent_c = Counter(trn_ints).most_common(10)
    print(slots_len)
    print(intent_len)
    print(slots_c)
    print(intent_c)
    slens = [len(s) for s in trn_texts]
    # text 平均长度
    print(np.mean(slens))
    for i in range(1, 3):
        print('txt:', len(trn_texts[-i]), trn_texts[-i])
        print('ent:', len(trn_slots[-i]), trn_slots[-i])
        print('int:', trn_ints[-i])
        print()
    # first, remove the BOS and EOS tags from the training sentences
    w2v_text = [s[1:-1] for s in trn_texts]
    # train and save model
    model = Word2Vec(w2v_text, size=200, min_count=1, window=5, workers=3, iter=5)
    model.save('model/atis_w2v.gensimmodel')
    print('training done!')
    vocab = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    model.wv.most_similar('delta')

    # instantiate a sentence indexer and fit to the training data
    sentindexer = CharacterIndexer(max_sent_mode='std')
    sentindexer.fit(trn_texts, verbose=True)
    # transform the sentence data
    trn_text_idx, trn_char_idx = sentindexer.transform(trn_texts)
    dev_text_idx, dev_char_idx = sentindexer.transform(dev_texts)
    tst_text_idx, tst_char_idx = sentindexer.transform(tst_texts)
    print(trn_text_idx.shape, dev_text_idx.shape, tst_text_idx.shape, trn_char_idx.shape)
    # instantiate a slot indexer and fit to the training data
    slotindexer = SlotIndexer(max_len=sentindexer.max_sent_len)
    slotindexer.fit(trn_slots, verbose=True)
    # transform the slot data
    trn_slot_idx = slotindexer.transform(trn_slots)
    dev_slot_idx = slotindexer.transform(dev_slots)
    tst_slot_idx = slotindexer.transform(tst_slots)
    print(trn_slot_idx.shape, dev_slot_idx.shape, tst_slot_idx.shape)
    intindexer = IntentIndexer()
    intindexer.fit(trn_ints, verbose=True)
    # transform the intent data
    trn_int_idx = intindexer.transform(trn_ints)
    dev_int_idx = intindexer.transform(dev_ints)
    tst_int_idx = intindexer.transform(tst_ints)
    print(trn_int_idx.shape, dev_int_idx.shape, tst_int_idx.shape)

    print(np.unique(np.isnan(trn_text_idx)))
    print(np.unique(np.isnan(trn_char_idx)))
    print(np.unique(np.isnan(dev_text_idx)))
    print(np.unique(np.isnan(dev_char_idx)))
    print(np.unique(np.isnan(tst_text_idx)))
    print(np.unique(np.isnan(tst_char_idx)))

    print(np.unique(np.isnan(trn_slot_idx)))
    print(np.unique(np.isnan(dev_slot_idx)))
    print(np.unique(np.isnan(tst_slot_idx)))

    print(np.unique(np.isnan(trn_int_idx)))
    print(np.unique(np.isnan(dev_int_idx)))
    print(np.unique(np.isnan(tst_int_idx)))

    print(sentindexer.max_word_vocab)
    print(np.unique(np.max(trn_text_idx)))
    print(np.unique(np.max(dev_text_idx)))
    print(np.unique(np.max(tst_text_idx)))
    print(sentindexer.max_char_vocab)
    print(np.unique(np.max(trn_char_idx)))
    print(np.unique(np.max(dev_char_idx)))
    print(np.unique(np.max(tst_char_idx)))
    print(slotindexer.labelsize)
    print(np.unique(np.max(trn_slot_idx)))
    print(np.unique(np.max(dev_slot_idx)))
    print(np.unique(np.max(tst_slot_idx)))
    print(intindexer.labelsize)
    print(np.unique(np.max(trn_int_idx)))
    print(np.unique(np.max(dev_int_idx)))
    print(np.unique(np.max(tst_int_idx)))

    print(sentindexer.inverse_transform(tst_text_idx[0:1]))

    print(slotindexer.inverse_transform(tst_slot_idx[0:1]))
    print(intindexer.inverse_transform(tst_int_idx[0:5]))

    # save transformers
    pickle.dump(sentindexer, open('encoded/atis_sentindexer.pkl', 'wb'))
    pickle.dump(slotindexer, open('encoded/atis_slotindexer.pkl', 'wb'))
    pickle.dump(intindexer, open('encoded/atis_intindexer.pkl', 'wb'))

    # save word2vec vocab
    pickle.dump(vocab, open('model/atis_w2v_vocab.pkl', 'wb'))

    # save text data
    pickle.dump(trn_texts, open('encoded/trn_texts_raw.pkl', 'wb'))
    pickle.dump(dev_texts, open('encoded/dev_texts_raw.pkl', 'wb'))
    pickle.dump(tst_texts, open('encoded/tst_texts_raw.pkl', 'wb'))

    pickle.dump(trn_slots, open('encoded/trn_slots_raw.pkl', 'wb'))
    pickle.dump(dev_slots, open('encoded/dev_slots_raw.pkl', 'wb'))
    pickle.dump(tst_slots, open('encoded/tst_slots_raw.pkl', 'wb'))

    pickle.dump(trn_ints, open('encoded/trn_ints_raw.pkl', 'wb'))
    pickle.dump(dev_ints, open('encoded/dev_ints_raw.pkl', 'wb'))
    pickle.dump(tst_ints, open('encoded/tst_ints_raw.pkl', 'wb'))

    # save encoded data
    np.save('encoded/trn_text_idx.npy', trn_text_idx)
    np.save('encoded/dev_text_idx.npy', dev_text_idx)
    np.save('encoded/tst_text_idx.npy', tst_text_idx)

    np.save('encoded/trn_char_idx.npy', trn_char_idx)
    np.save('encoded/dev_char_idx.npy', dev_char_idx)
    np.save('encoded/tst_char_idx.npy', tst_char_idx)

    np.save('encoded/trn_slot_idx.npy', trn_slot_idx)
    np.save('encoded/dev_slot_idx.npy', dev_slot_idx)
    np.save('encoded/tst_slot_idx.npy', tst_slot_idx)

    np.save('encoded/trn_int_idx.npy', trn_int_idx)
    np.save('encoded/dev_int_idx.npy', dev_int_idx)
    np.save('encoded/tst_int_idx.npy', tst_int_idx)