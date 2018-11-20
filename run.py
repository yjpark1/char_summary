import numpy as np
import pandas as pd
import keras
from keras.preprocessing import text
from models.attention_seq2seq import seq2seq_attention
from train.TrainWithLabeledText import (make_target_input,
                                        make_target_output,
                                        train_valid_split,
                                        DataGenerator)
from tqdm import tqdm

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


flag_train = True
type_inference = 'beamsearch'
valid_index = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

text_morph = np.load('datasets/text.npy').tolist()
summary_morph = np.load('datasets/summary.npy').tolist()

# make sequence: text
t_txt = text.Tokenizer(filters='')
t_txt.fit_on_texts(text_morph)
txt = t_txt.texts_to_sequences(text_morph)

# make sequence: summary
t_summ = text.Tokenizer(filters='')
t_summ.fit_on_texts(summary_morph)
summ = t_summ.texts_to_sequences(summary_morph)

# make summary output dataset
summ_input = make_target_input(summ)
summ_output = make_target_output(summ)

# split train/test
train_dataset, valid_dataset = train_valid_split(txt, summ_input, summ_output, valid_index)
txt, summ_input, summ_output = train_dataset
val_txt, val_summ_input, val_summ_output = valid_dataset
# generator
gen = DataGenerator(text=txt, summary_input=summ_input,
                    summary_target=summ_output,
                    num_token_output=len(t_summ.index_word),
                    idx_txt_split=t_txt.word_index['.'],
                    batch_size=16)
gen.to_multi_sentence()
gen.sort_data()

val_gen = DataGenerator(text=val_txt, summary_input=val_summ_input,
                        summary_target=val_summ_output,
                        num_token_output=len(t_summ.index_word),
                        idx_txt_split=t_txt.word_index['.'],
                        batch_size=10)
val_gen.to_multi_sentence()
val_gen.sort_data()

# train model
model = seq2seq_attention(num_encoder_tokens=len(t_txt.index_word), embedding_dim=64,
                          hidden_dim=128, num_decoder_tokens=len(t_summ.index_word),
                          input_tokenizer=t_txt, target_tokenizer=t_summ)
summaryModel = model.get_model()
summaryModel.compile(optimizer='Adam', loss='categorical_crossentropy')
if flag_train:
    checkpoint = keras.callbacks.ModelCheckpoint('results/seq2seq_atten.h5',
                                                 monitor='val_loss', save_best_only=True,
                                                 save_weights_only=True,
                                                 mode='min')
    earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min',
                                              restore_best_weights=True)

    summaryModel.fit_generator(generator=gen,
                               validation_data=val_gen,
                               epochs=150, use_multiprocessing=True,
                               workers=2, verbose=2, callbacks=[checkpoint, earlystop])

    summaryModel.save_weights('results/seq2seq_atten.h5')

else:
    summaryModel.load_weights('results/seq2seq_atten.h5')

model.summaryModel = summaryModel

# build inference model
model.build_inference_model()

# run inference
print('start inference...')

pred = []
for i in tqdm(range(len(text_morph))):
    if type_inference == 'beamsearch':
        p = model.inference_beamsearch(input_text=text_morph[i])
        p = p[0][np.argmax(p[1])]

    elif type_inference == 'greedy':
        p = model.inference_greedy(input_text=text_morph[i])
        p = p[0]

    pred.append(p)

np.save('results/pred_' + type_inference + '.npy', pred)

candidate = np.load('results/pred_' + type_inference + '.npy').tolist()
pred = np.array([x for x in candidate])

re = pd.DataFrame({'true': summary_morph, 'pred': pred, 'text': text_morph})
re.to_csv('results/result_seq2seq_attention_' + type_inference + '.csv', encoding='cp949')