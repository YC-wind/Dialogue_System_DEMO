#!/usr/bin/python
# coding:utf-8
"""
@author: Cong Yu

@time: 2019/1/10 2:37 PM
"""
from collections import Counter
from preprocessing import CharacterIndexer, SlotIndexer, IntentIndexer
from gensim.models import Word2Vec
import json
import numpy as np
import pandas as pd
import pickle

sentindexer = pickle.load(open('encoded/atis_sentindexer.pkl', 'rb'))
slotindexer = pickle.load(open('encoded/atis_slotindexer.pkl', 'rb'))
intindexer  = pickle.load(open('encoded/atis_intindexer.pkl',  'rb'))

trn_text_idx = np.load('encoded/trn_text_idx.npy')
trn_char_idx = np.load('encoded/trn_char_idx.npy')
trn_slot_idx = np.load('encoded/trn_slot_idx.npy')
trn_int_idx  = np.load('encoded/trn_int_idx.npy')

dev_text_idx = np.load('encoded/dev_text_idx.npy')
dev_char_idx = np.load('encoded/dev_char_idx.npy')
dev_slot_idx = np.load('encoded/dev_slot_idx.npy')
dev_int_idx  = np.load('encoded/dev_int_idx.npy')

tst_text_idx = np.load('encoded/tst_text_idx.npy')
tst_char_idx = np.load('encoded/tst_char_idx.npy')
tst_slot_idx = np.load('encoded/tst_slot_idx.npy')
tst_int_idx  = np.load('encoded/tst_int_idx.npy')

w2v_model = Word2Vec.load('model/atis_w2v.gensimmodel')
w2v_vocab = pickle.load(open('model/atis_w2v_vocab.pkl',  'rb'))

import h5py
import math
from keras.models import Model
from keras.layers import Activation, Concatenate, concatenate, Dense, Dropout, Embedding, Input, TimeDistributed
from keras.layers import LSTM, CuDNNLSTM, LeakyReLU, Masking, Lambda, Dot, BatchNormalization, Activation
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Flatten
from keras.layers.wrappers import Bidirectional
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TerminateOnNaN, ModelCheckpoint
from keras_contrib.layers import CRF
from keras_contrib.utils import save_load_utils

from keras_utilities import AttentionWithContext

from keras.optimizers import Adam, SGD
import keras.backend as K
from keras.layers import Dense, Activation, Multiply, Add, Lambda
import keras.initializers
from keras.regularizers import l1, l2

"""
the joint model is based on Hakkani-Tur with improvements:

- optional slot CRF following Ma & Hovy for NER
- optional "aligned seq2seq" following Liu & Lane (state initialization in deep LSTM)
- optional pre-slot output attention following Liu & Lane
- optional merging of slot predictions + attention for intent following Lui & Lane 

### implementation notes

- the highway bias is set to -2 as recommended by Yoon Kim  
- the activation is set to `relu` following the article, although the sample code uses `relu`  
- the optimizer is set to `adadelta` as it seems to converge well. `clipnorm` is used because of a `nan` loss problem  
- due to the above `nan` loss issue, there is some slight regularization applied to the intent dense layers.
"""

modelname = 'test_model'
# preprocessing-dependent parameters
# we can use the indexer attributes
TXT_VOCAB  = sentindexer.max_word_vocab
TXT_MAXLEN = sentindexer.max_sent_len
CHR_MAXLEN = sentindexer.max_word_len
CHR_VOCAB  = sentindexer.max_char_vocab
SLOT_NUM   = slotindexer.labelsize
LABEL_NUM  = intindexer.labelsize
print(TXT_VOCAB, TXT_MAXLEN, SLOT_NUM, LABEL_NUM)

# self-defined network hyperparameters
WEMBED_SIZE   = 200   # word embedding size. must match w2v size
CEMBED_SIZE   = 200   # character embedding size. free param
WDROP_RATE    = 0.50  # word-level input dropout
DROP_RATE     = 0.33  # dropout for other layers
RNN_DROP_RATE = 0.0   # recurrent droput (not implemented)
HIDDEN_SIZE   = 300   # LSTM block hidden size
BATCH_SIZE    = 32
MAX_EPOCHS    = 50
OPTIMIZER     = keras.optimizers.Adadelta(clipnorm=1.)

########################################
# preload word vectors
########################################

# create word embedding matrix
# load word2vec vector if present; otherwise randomly init, but keep padding zero
# ref: https://github.com/UKPLab/emnlp2017-bilstm-cnn-crf/blob/master/neuralnets/BiLSTM.py
word_embedding_matrix = np.zeros((TXT_VOCAB, WEMBED_SIZE))
c = 0
for w in sentindexer.word2idx.keys():
    # get the word vector from the embedding model
    # if it's there (check against vocab list)
    if w in w2v_vocab:
        # get the word vector
        word_vector = w2v_model.wv[w]
        # slot it in at the proper index
        word_embedding_matrix[sentindexer.word2idx[w]] = word_vector
        c += 1
    elif w not in ("PAD", "_PAD_"):
        limit = math.sqrt(3.0 / WEMBED_SIZE)
        word_vector = np.random.uniform(-limit, limit, WEMBED_SIZE)
        word_embedding_matrix[sentindexer.word2idx[w]] = word_vector


# loaded vector # may be lower than total vocab due to w2v settings
print('loaded total of', c, 'vectors')
########################################
# randomly init char vectors
########################################


# create char embedding matrix randomly but keep padding zero
# ref: https://github.com/UKPLab/emnlp2017-bilstm-cnn-crf/blob/master/neuralnets/BiLSTM.py
char_embedding_matrix = np.zeros((CHR_VOCAB, CEMBED_SIZE))
c = 0
for w in sentindexer.char2idx.keys():
    if w not in ("PAD", "_PAD_"):
        # slot it in at the proper index
        limit = math.sqrt(3.0 / CEMBED_SIZE)
        char_vector = np.random.uniform(-limit, limit, CEMBED_SIZE)
        char_embedding_matrix[sentindexer.char2idx[w]] = char_vector
        c += 1


# loaded vector # may be lower than total vocab due to w2v settings
print('initialized total of', c, 'vectors')

########################################
# Kim; Ma & Hovy char-CNN + word input
########################################

# word-level input with word embedding matrix (with word2vec)
txt_input = Input(shape=(TXT_MAXLEN,), name='word_input')

txt_embed = Embedding(TXT_VOCAB, WEMBED_SIZE, input_length=TXT_MAXLEN,
                      weights=[word_embedding_matrix],
                      name='word_embedding', trainable=True, mask_zero=True)(txt_input)

txt_drpot = Dropout(WDROP_RATE, name='word_dropout')(txt_embed)

# character-level input with randomized initializations
cnn_input = Input(shape=(TXT_MAXLEN, CHR_MAXLEN), name='cnn_input')

cnn_embed = TimeDistributed(Embedding(CHR_VOCAB, CEMBED_SIZE, input_length=CHR_MAXLEN,
                            weights=[char_embedding_matrix],
                            name='cnn_embedding', trainable=True, mask_zero=False))(cnn_input)

# 1-size window CNN with batch-norm & tanh activation (Kim 2015)
cnns1 = TimeDistributed(Conv1D(filters=20, kernel_size=1, padding="same", strides=1), name='cnn1_cnn')(cnn_embed)
cnns1 = TimeDistributed(BatchNormalization(), name='cnn1_bnorm')(cnns1)
cnns1 = TimeDistributed(Activation('tanh'), name='cnn1_act')(cnns1)
cnns1 = TimeDistributed(GlobalMaxPooling1D(), name='cnn1_gmp')(cnns1)

# 2-size window CNN with batch-norm & tanh activation (Kim 2015)
cnns2 = TimeDistributed(Conv1D(filters=40, kernel_size=2, padding="same", strides=1), name='cnn2_cnn')(cnn_embed)
cnns2 = TimeDistributed(BatchNormalization(), name='cnn2_bnorm')(cnns2)
cnns2 = TimeDistributed(Activation('tanh'), name='cnn2_act')(cnns2)
cnns2 = TimeDistributed(GlobalMaxPooling1D(), name='cnn2_gmp')(cnns2)

# 3-size window CNN with batch-norm & tanh activation (Kim 2015)
cnns3 = TimeDistributed(Conv1D(filters=60, kernel_size=3, padding="same", strides=1), name='cnn3_cnn')(cnn_embed)
cnns3 = TimeDistributed(BatchNormalization(), name='cnn3_bnorm')(cnns3)
cnns3 = TimeDistributed(Activation('tanh'), name='cnn3_act')(cnns3)
cnns3 = TimeDistributed(GlobalMaxPooling1D(), name='cnn3_gmp')(cnns3)

# 4-size window CNN with batch-norm & tanh activation (Kim 2015)
cnns4 = TimeDistributed(Conv1D(filters=80, kernel_size=4, padding="same", strides=1), name='cnn4_cnn')(cnn_embed)
cnns4 = TimeDistributed(BatchNormalization(), name='cnn4_bnorm')(cnns4)
cnns4 = TimeDistributed(Activation('tanh'), name='cnn4_act')(cnns4)
cnns4 = TimeDistributed(GlobalMaxPooling1D(), name='cnn4_gmp')(cnns4)

# time-distributed highway layer (Kim 2015)
cnns  = concatenate([cnns1, cnns2, cnns3, cnns4], axis=-1, name='cnn_concat')

K.int_shape(cnns)[-1] # 200
"""
### highway layer

paraphrasing from Yoon Kim(?) : "an extension of the LSTM network to feed-forward networks"

see:  
https://arxiv.org/pdf/1505.00387.pdf  
http://people.idsia.ch/~rupesh/very_deep_learning/  
https://theneuralperspective.com/2016/12/13/highway-networks/

coded following Srivastava et al. with reference to https://gist.github.com/iskandr/a874e4cf358697037d14a17020304535
"""
########################################
# subword vector highway layer
########################################

hway_input = Input(shape=(K.int_shape(cnns)[-1],))
gate_bias_init = keras.initializers.Constant(-2)
transform_gate = Dense(units=K.int_shape(cnns)[-1], bias_initializer=gate_bias_init, activation='sigmoid')(hway_input)
carry_gate = Lambda(lambda x: 1.0 - x, output_shape=(K.int_shape(cnns)[-1],))(transform_gate)
h_transformed = Dense(units=K.int_shape(cnns)[-1])(hway_input)
h_transformed = Activation('relu')(h_transformed)
transformed_gated = Multiply()([transform_gate, h_transformed])
carried_gated = Multiply()([carry_gate, hway_input])
outputs = Add()([transformed_gated, carried_gated])

highway = Model(inputs=hway_input, outputs=outputs)

cnns  = TimeDistributed(highway, name='cnn_highway')(cnns)

# final concat of convolutional subword embeddings and word vectors
word_vects  = concatenate([cnns, txt_drpot], axis=-1, name='concat_word_vectors')

########################################
# main recurrent sentence block
########################################

# 'encoder' layer with returned states following (Liu, Lane)
lstm_enc, fh, fc, bh, bc  = Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True, return_state=True),
                                          name='bidirectional_enc')(word_vects)
lstm_enc = Dropout(DROP_RATE, name='bidirectional_dropout_enc')(lstm_enc)

# "aligned seq2seq" lstm
# load forward LSTM with reverse states following Liu, Lane 2016 (and do reverse)
lstm_dec = Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True),
                         name='bidirectional_dec')(lstm_enc, initial_state=[bh, bc, fh, fc])

lstm_states = Dropout(DROP_RATE, name='bidirectional_dropout_dec')(lstm_dec)

########################################
# Huang et al; Ma & Hovy CRF slot clf
########################################

# final slot linear chain CRF layer
lyr_crf   = CRF(SLOT_NUM, sparse_target=True, name='out_slot', learn_mode='marginal', test_mode='marginal')
out_slot  = lyr_crf(lstm_states)

# alternative is using greedy predictions
# out_slot  = TimeDistributed(Dense(SLOT_NUM, activation='softmax'), name='out_slot')(txt_lstm_dec)

########################################
# attentional intent clf block
########################################

# combine lstm with CRF for attention (see Liu & Lane)
seq_concat = concatenate([lstm_states, out_slot], axis=2, name='lstm_concat')
seq_concat = Dropout(DROP_RATE, name='bidirectional_dropout_3')(seq_concat)

# layer: intent attention w/context (Liu & Lane)
att_int = AttentionWithContext(name='intent_attention')(seq_concat)

# layer: dense + LeakyReLU with dropout
out_int = Dense(K.int_shape(att_int)[-1],
                kernel_regularizer=l2(0.0025),
                name='intent_dense_1')(att_int)
out_int = LeakyReLU(name='intent_act_1')(out_int)
out_int = Dropout(DROP_RATE, name='intent_dropout_1')(out_int)

# layer: dense + LeakyReLU with dropout
out_int = Dense(K.int_shape(att_int)[-1],
                kernel_regularizer=l2(0.001),
                name='intent_dense_2')(out_int)
out_int = LeakyReLU(name='intent_act_2')(out_int)

# layer: final dense + softmax
out_int = Dense(LABEL_NUM, activation='softmax', name='out_intent')(out_int)

model = Model(inputs=[txt_input, cnn_input], outputs=[out_slot, out_int])

model.summary()

model.compile(optimizer=OPTIMIZER,
              loss={'out_slot': lyr_crf.loss_function, 'out_intent': 'sparse_categorical_crossentropy'},
              # loss={'out_slot': 'sparse_categorical_crossentropy', 'out_intent': 'sparse_categorical_crossentropy'},
              loss_weights={'out_slot': 0.5, 'out_intent': 0.5},
              )
# callbacks
# cb_redlr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=2, min_lr=0.0001, verbose=1)
cb_early = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1)
cb_chkpt = ModelCheckpoint('checkpoints/_'+modelname+'{epoch:02d}-{val_loss:.2f}.h5', verbose=1, save_best_only=True, save_weights_only=True, period=5)
cb_nonan = TerminateOnNaN()

callbacks_list=[cb_early, cb_chkpt, cb_nonan]

# # load weights to resume training
# model.load_weights('checkpoints/FILE_NAME_HERE.h5')


history = model.fit([trn_text_idx, trn_char_idx],
                    [trn_slot_idx, trn_int_idx],
                    validation_data=([dev_text_idx, dev_char_idx], [dev_slot_idx, dev_int_idx]),
                    batch_size=BATCH_SIZE,
                    epochs=MAX_EPOCHS,
                    callbacks=callbacks_list,
                    verbose=0)

hist_dict = history.history

# save architecture with json
with open('model/'+modelname+'.json', 'w') as f:
    f.write(model.to_json())
# save weights
save_load_utils.save_all_weights(model, 'model/'+modelname+'.h5')
# save training history
np.save('model/'+modelname+'_dict.npy', hist_dict)

# load test
model.load_weights('model/'+modelname+'.h5')

from sklearn.metrics import classification_report, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score

# remove nulls and pads and get F1 on only labels
def procslots(trues, preds, nonull=True):
    tru_slots = []
    prd_slots = []
    for i in range(len(trues)):
        for j in range(len(trues[i])):
            tru = trues[i][j]
            if j < len(preds[i]):
                prd = preds[i][j]
            else:
                prd = 'O'
            if nonull==True:
                if tru not in ('O', slotindexer.pad, slotindexer.unk):
                    tru_slots.append(tru)
                    prd_slots.append(prd)
            else:
                tru_slots.append(tru)
                prd_slots.append(prd)
    return tru_slots, prd_slots

tprd_slots_dist, tprd_ints_dist = model.predict([trn_text_idx, trn_char_idx])
tprd_int_idx  = np.squeeze(np.argmax(tprd_ints_dist, axis=-1))

tprd_slot_idx = np.argmax(tprd_slots_dist, axis=-1)
tprd_ints = intindexer.inverse_transform(tprd_int_idx)
ttru_ints = intindexer.inverse_transform(trn_int_idx)

# convert slot predictions, trues to text form
tprd_txtslots = slotindexer.inverse_transform(tprd_slot_idx)
ttrn_txtslots = slotindexer.inverse_transform(trn_slot_idx)

ttru_slots, tprd_slots = procslots(tprd_txtslots, ttrn_txtslots, nonull=True)

print('# TRAIN RESULTS')
print()
print('INTENT F1 :  ', f1_score(ttru_ints, tprd_ints, average='weighted'), ' (weighted)')
print('INTENT ACC:  ', accuracy_score(ttru_ints, tprd_ints))
print('SLOT   F1 :  ', f1_score(ttru_slots, tprd_slots, average='weighted'), ' (weighted, labels only)')

prd_slots_dist, prd_ints_dist = model.predict([tst_text_idx, tst_char_idx])
prd_int_idx  = np.squeeze(np.argmax(prd_ints_dist, axis=-1))

prd_slot_idx = np.argmax(prd_slots_dist, axis=-1)

prd_ints = intindexer.inverse_transform(prd_int_idx)
tru_ints = intindexer.inverse_transform(tst_int_idx)

# convert slot predictions, trues to text form
prd_txtslots = slotindexer.inverse_transform(prd_slot_idx)
tst_txtslots = slotindexer.inverse_transform(tst_slot_idx)

tru_slots, prd_slots = procslots(prd_txtslots, tst_txtslots, nonull=True)

print('# TEST RESULTS')
print()
print('INTENT F1 :  ', f1_score(tru_ints, prd_ints, average='weighted'), ' (weighted)')
print('INTENT ACC:  ', accuracy_score(tru_ints, prd_ints))
print('SLOT   F1 :  ', f1_score(tru_slots, prd_slots, average='weighted'), ' (weighted, labels only)')


if __name__ == "__main__":

    print()