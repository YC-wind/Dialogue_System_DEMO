#说明

具体参考 
https://github.com/SNUDerek/multiLSTM

```
gensim==3.4.0
h5py==2.8.0
Keras==2.2.0
keras-contrib==2.0.8
keras-utilities==0.5.0
numpy
tensorflow==1.9.0
```

## 模型结构

the model accepts two inputs representing one sentence: a 1D array of integer-indexed **word-level** tokenized input, e.g. [hello, world, this, is, a, test] and 2D array of per-word **character-level** input: [[h,e,l,l,o], [w,or,l,d],...,[t,e,s,t]]

Ma & Hovy 2016 note that research has found that using **multiple input methods**, e.g. distributed word embeddings & engineered features, outperform **single input method** models.

we may hope that the **word-level embeddings**, trained via context, can detect a mixture of **syntactic and semantic** features, e.g. scholar has a high degree of 'nouniness' and appears related to other academic words; 

while the **character-level embeddings**, by using convolutional networks of various widths, may focus on detecting **n-gram patterns** such as 'schola' that may relate the word to words with similar patterns such as 'scholastic', 'scholarship' etc. 

this may help the network recognize **unseen words** by their subword features alone.

we also pass the subword embeddings through **a highway layer**, as proposed by Yoon Kim in his character-based CNN-LSTM language model as an alternative to a simple feed-forward network (or none at all).

this layer adds a **gating function** that controls information flow and is roughly analogous to things like the **resnet architecture**, in that it is primarily used for training very deep networks. it's included here because of the essential ml research principle, the Rule of Cool.

```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
cnn_input (InputLayer)          (None, 22, 9)        0                                            
__________________________________________________________________________________________________
time_distributed_1 (TimeDistrib (None, 22, 9, 200)   7000        cnn_input[0][0]                  
__________________________________________________________________________________________________
cnn1_cnn (TimeDistributed)      (None, 22, 9, 20)    4020        time_distributed_1[0][0]         
__________________________________________________________________________________________________
cnn2_cnn (TimeDistributed)      (None, 22, 9, 40)    16040       time_distributed_1[0][0]         
__________________________________________________________________________________________________
cnn3_cnn (TimeDistributed)      (None, 22, 9, 60)    36060       time_distributed_1[0][0]         
__________________________________________________________________________________________________
cnn4_cnn (TimeDistributed)      (None, 22, 9, 80)    64080       time_distributed_1[0][0]         
__________________________________________________________________________________________________
cnn1_bnorm (TimeDistributed)    (None, 22, 9, 20)    80          cnn1_cnn[0][0]                   
__________________________________________________________________________________________________
cnn2_bnorm (TimeDistributed)    (None, 22, 9, 40)    160         cnn2_cnn[0][0]                   
__________________________________________________________________________________________________
cnn3_bnorm (TimeDistributed)    (None, 22, 9, 60)    240         cnn3_cnn[0][0]                   
__________________________________________________________________________________________________
cnn4_bnorm (TimeDistributed)    (None, 22, 9, 80)    320         cnn4_cnn[0][0]                   
__________________________________________________________________________________________________
cnn1_act (TimeDistributed)      (None, 22, 9, 20)    0           cnn1_bnorm[0][0]                 
__________________________________________________________________________________________________
cnn2_act (TimeDistributed)      (None, 22, 9, 40)    0           cnn2_bnorm[0][0]                 
__________________________________________________________________________________________________
cnn3_act (TimeDistributed)      (None, 22, 9, 60)    0           cnn3_bnorm[0][0]                 
__________________________________________________________________________________________________
cnn4_act (TimeDistributed)      (None, 22, 9, 80)    0           cnn4_bnorm[0][0]                 
__________________________________________________________________________________________________
cnn1_gmp (TimeDistributed)      (None, 22, 20)       0           cnn1_act[0][0]                   
__________________________________________________________________________________________________
cnn2_gmp (TimeDistributed)      (None, 22, 40)       0           cnn2_act[0][0]                   
__________________________________________________________________________________________________
cnn3_gmp (TimeDistributed)      (None, 22, 60)       0           cnn3_act[0][0]                   
__________________________________________________________________________________________________
cnn4_gmp (TimeDistributed)      (None, 22, 80)       0           cnn4_act[0][0]                   
__________________________________________________________________________________________________
word_input (InputLayer)         (None, 22)           0                                            
__________________________________________________________________________________________________
cnn_concat (Concatenate)        (None, 22, 200)      0           cnn1_gmp[0][0]                   
                                                                 cnn2_gmp[0][0]                   
                                                                 cnn3_gmp[0][0]                   
                                                                 cnn4_gmp[0][0]                   
__________________________________________________________________________________________________
word_embedding (Embedding)      (None, 22, 200)      145600      word_input[0][0]                 
__________________________________________________________________________________________________
cnn_highway (TimeDistributed)   (None, 22, 200)      80400       cnn_concat[0][0]                 
__________________________________________________________________________________________________
word_dropout (Dropout)          (None, 22, 200)      0           word_embedding[0][0]             
__________________________________________________________________________________________________
concat_word_vectors (Concatenat (None, 22, 400)      0           cnn_highway[0][0]                
                                                                 word_dropout[0][0]               
__________________________________________________________________________________________________
bidirectional_enc (Bidirectiona [(None, 22, 600), (N 1682400     concat_word_vectors[0][0]        
__________________________________________________________________________________________________
bidirectional_dropout_enc (Drop (None, 22, 600)      0           bidirectional_enc[0][0]          
__________________________________________________________________________________________________
bidirectional_dec (Bidirectiona (None, 22, 600)      2162400     bidirectional_dropout_enc[0][0]  
                                                                 bidirectional_enc[0][3]          
                                                                 bidirectional_enc[0][4]          
                                                                 bidirectional_enc[0][1]          
                                                                 bidirectional_enc[0][2]          
__________________________________________________________________________________________________
bidirectional_dropout_dec (Drop (None, 22, 600)      0           bidirectional_dec[0][0]          
__________________________________________________________________________________________________
out_slot (CRF)                  (None, 22, 121)      87604       bidirectional_dropout_dec[0][0]  
__________________________________________________________________________________________________
lstm_concat (Concatenate)       (None, 22, 721)      0           bidirectional_dropout_dec[0][0]  
                                                                 out_slot[0][0]                   
__________________________________________________________________________________________________
bidirectional_dropout_3 (Dropou (None, 22, 721)      0           lstm_concat[0][0]                
__________________________________________________________________________________________________
intent_attention (AttentionWith (None, 721)          521283      bidirectional_dropout_3[0][0]    
__________________________________________________________________________________________________
intent_dense_1 (Dense)          (None, 721)          520562      intent_attention[0][0]           
__________________________________________________________________________________________________
intent_act_1 (LeakyReLU)        (None, 721)          0           intent_dense_1[0][0]             
__________________________________________________________________________________________________
intent_dropout_1 (Dropout)      (None, 721)          0           intent_act_1[0][0]               
__________________________________________________________________________________________________
intent_dense_2 (Dense)          (None, 721)          520562      intent_dropout_1[0][0]           
__________________________________________________________________________________________________
intent_act_2 (LeakyReLU)        (None, 721)          0           intent_dense_2[0][0]             
__________________________________________________________________________________________________
out_intent (Dense)              (None, 22)           15884       intent_act_2[0][0]               
==================================================================================================
Total params: 5,864,695
Trainable params: 5,864,295
Non-trainable params: 400
__________________________________________________________________________________________________
```
## 结果

```
# TRAIN RESULTS

INTENT F1 :   0.9958062755602081  (weighted)
INTENT ACC:   0.9966502903081733
SLOT   F1 :   0.9943244981289089  (weighted, labels only)

# TEST RESULTS

INTENT F1 :   0.9573853410886499  (weighted)
INTENT ACC:   0.9630459126539753
SLOT   F1 :   0.9508826933073504  (weighted, labels only)
CONLLEVAL :   93.71
```

## paper

Goo et al (2018): Slot-Gated Modeling for Joint Slot Filling and Intent Prediction
NAACL-HCT 2018, available: http://aclweb.org/anthology/N18-2118

Hakkani-Tur et al (2016): Multi-Domain Joint Semantic Frame Parsing using Bi-directional RNN-LSTM
available: https://www.csie.ntu.edu.tw/~yvchen/doc/IS16_MultiJoint.pdf

Kim et al (2015): Character-Aware Neural Language Models
available: https://arxiv.org/pdf/1508.06615.pdf

Liu & Lane (2016): Attention-Based Recurrent Neural Network Models for Joint Intent Detection and Slot Filling
INTERSPEECH 2016, available: https://pdfs.semanticscholar.org/84a9/bc5294dded8d597c9d1c958fe21e4614ff8f.pdf

Ma & Hovy (2016): End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF
available: https://arxiv.org/pdf/1603.01354.pdf

Park & Song (2017): 음절 기반의 CNN 를 이용한 개체명 인식 Named Entity Recognition using CNN for Korean syllabic character
available: https://www.dbpia.co.kr/Journal/ArticleDetail/NODE07017625 (Korean)

Srivastava, R. K., Greff, K., & Schmidhuber, J. (2015). Highway networks.
available: https://arxiv.org/pdf/1505.00387.pdf