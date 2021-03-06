import numpy as np
import keras
from keras.preprocessing import text, sequence
from models.attention_seq2seq import seq2seq_attention


def make_target_input(target):
    target = [x[:-1] for x in target]
    return target


def make_target_output(target):
    target = [x[1:] for x in target]
    return target


def make_onehot_target(target, num_token_output):
    out = []
    for i, x in enumerate(target):
        x = np.array(x) - 1
        out.append(np.eye(num_token_output)[x])
    return out


# input sequence length
# seq_len = [len(x) for x in txt]
# print('mean of input sequence length: ', np.mean(seq_len))  # 756.1045454545455
# print('max of input sequence length: ', np.max(seq_len))  # 7959
# print('min of input sequence length: ', np.min(seq_len))  # 73
# need dynamic rnn!!

# revert example: sequence to text
# t_summ.sequences_to_texts(summ[:2])

class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""
    def __init__(self, text, summary_input, summary_target,
                 num_token_output, idx_txt_split, batch_size=16):
        """Initialization"""
        self.text = np.array(text)
        self.summary_input = np.array(summary_input)
        self.summary_target = np.array(summary_target)
        self.batch_size = batch_size
        self.shuffle = False
        self.indexes = np.arange(len(text))
        self.num_token_output = num_token_output
        self.idx_txt_split = idx_txt_split

    def to_multi_sentence(self):
        out = []
        for x in self.text:
            x = np.array(x)
            split_point = np.where(x == self.idx_txt_split)[0]
            x = np.array(np.split(x, split_point + 1))
            idx = np.array([len(a) for a in x]) > 1
            out.append(x[idx])
        self.text = np.array(out)

    def sort_data(self):
        # sort by length: text, summary_input, summary_target
        text_length = [len(x) for x in self.text]
        index = np.argsort(text_length)

        self.text = self.text[index]
        self.summary_input = self.summary_input[index]
        self.summary_target = self.summary_target[index]

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.text) / self.batch_size))

    def __make_onehot_target(self, target):
        out = []
        for i, x in enumerate(target):
            x = np.array(x) - 1
            out.append(np.eye(self.num_token_output)[x])
        return out

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def pad_sentence(self, x, maxlen):
        shp = x.shape
        add = maxlen - shp[0]
        if add > 0:
            add = np.zeros(shape=(add, shp[-1]))
            x = np.concatenate([x, add], axis=0)
        return x

    def __data_generation(self, indexes):
        """Generates data containing batch_size samples
        # X : (n_samples, *dim, n_channels)
        """
        # get batch data
        batch_text = self.text[indexes]
        batch_summary_input = self.summary_input[indexes]
        batch_summary_target = self.summary_target[indexes]

        max_sen_txt = np.max([len(x) for x in batch_text])
        max_char_txt = np.max([np.max([len(x) for x in doc]) for doc in batch_text])
        max_char_txt.flatten()
        max_len_summ_input = np.max([len(x) for x in batch_summary_input])

        # preprocessing
        # 1) make onehot target
        batch_summary_target = self.__make_onehot_target(batch_summary_target)

        # 2) pad sequence
        ## pad char
        batch_text = [sequence.pad_sequences(x, maxlen=max_char_txt,
                                             truncating='post',
                                             padding='post') for x in batch_text]

        batch_text = [self.pad_sentence(x, maxlen=max_sen_txt) for x in batch_text]
        batch_text = np.array(batch_text)

        batch_summary_input = sequence.pad_sequences(batch_summary_input, maxlen=max_len_summ_input, padding='post')
        batch_summary_target = sequence.pad_sequences(batch_summary_target, maxlen=max_len_summ_input, padding='post')

        return [batch_text, batch_summary_input], batch_summary_target


# In future, validation generator will be used.
def train_valid_split(txt, summ_input, summ_output, valid_index):
    valid_index = np.array(valid_index)

    # make numpy array for indexing
    txt = np.array(txt)
    summ_input = np.array(summ_input)
    summ_output = np.array(summ_output)

    # make validation dataset
    valid_text = txt[valid_index].tolist()
    valid_summary_input = summ_input[valid_index].tolist()
    valid_summary_target = summ_output[valid_index].tolist()

    # remove validation dataset in training dataset
    txt = np.delete(txt, valid_index).tolist()
    summ_input = np.delete(summ_input, valid_index).tolist()
    summ_output = np.delete(summ_output, valid_index).tolist()

    train_dataset = (txt, summ_input, summ_output)
    valid_dataset = (valid_text, valid_summary_input, valid_summary_target)

    return train_dataset, valid_dataset


if __name__ == '__main__':
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
                            batch_size=16)
    val_gen.to_multi_sentence()
    val_gen.sort_data()

    # train model
    model = seq2seq_attention(num_encoder_tokens=len(t_txt.index_word), embedding_dim=64,
                              hidden_dim=128, num_decoder_tokens=len(t_summ.index_word),
                              input_tokenizer=t_txt, target_tokenizer=t_summ)
    summaryModel = model.get_model()
    summaryModel.compile(optimizer='Adam', loss='categorical_crossentropy')

    summaryModel.fit_generator(generator=gen,
                               validation_data=val_gen,
                               epochs=2, use_multiprocessing=True,
                               workers=2, verbose=2)






