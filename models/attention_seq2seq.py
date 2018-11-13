import numpy as np
import heapq
from keras import backend as K
from keras.layers import Input, Embedding, Bidirectional
from keras.layers import concatenate
from keras.layers import RNN, LSTM, GRUCell
from keras.layers import Dense, Lambda
from keras.models import Model
from model.custom_layers import DenseAnnotationAttention


class seq2seq_attention:
    def __init__(self, num_encoder_tokens, embedding_dim,
                 hidden_dim, num_decoder_tokens,
                 input_tokenizer, target_tokenizer):
        self.num_encoder_tokens = num_encoder_tokens
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_decoder_tokens = num_decoder_tokens
        self.input_tokenizer = input_tokenizer
        self.target_tokenizer = target_tokenizer
        self.summaryModel = None
        self.start_token = '<start>'
        self.end_token = '<stop>'


    def dense_maxout(self, x_):
        """Implements a dense maxout layer where max is taken
        over _two_ units"""
        x_ = Dense(self.hidden_dim)(x_)
        x_1 = x_[:, :self.hidden_dim // 2]
        x_2 = x_[:, self.hidden_dim // 2:]
        return K.max(K.stack([x_1, x_2], axis=-1), axis=-1, keepdims=False)


    def get_model(self):
        # Input text
        encoder_inputs = Input(shape=(None,), name='input_text')
        # Input summary
        decoder_inputs = Input(shape=(None,), name='input_summary')

        # word embedding layer for text
        encoder_inputs_emb = Embedding(input_dim=self.num_encoder_tokens,
                                       output_dim=self.embedding_dim,
                                       mask_zero=True,
                                       name='embedding_text')(encoder_inputs)
        # word embedding layer for summary
        decoder_inputs_emb = Embedding(input_dim=self.num_decoder_tokens,
                                       output_dim=self.embedding_dim,
                                       mask_zero=True,
                                       name='embedding_summary')(decoder_inputs)

        # Bidirectional LSTM encoder
        encoder_out = Bidirectional(LSTM(self.hidden_dim // 2,
                                         return_sequences=True,
                                         return_state=True),
                                    merge_mode='concat',
                                    name='encoder')(encoder_inputs_emb)

        encoder_o = encoder_out[0]
        initial_h_lstm = concatenate([encoder_out[1], encoder_out[2]])
        initial_c_lstm = concatenate([encoder_out[3], encoder_out[4]])
        initial_decoder_state = Dense(self.hidden_dim, activation='tanh', name='decoder_state')(concatenate([initial_h_lstm, initial_c_lstm]))

        # LSTM decoder + attention
        initial_attention_h = Lambda(lambda x: K.zeros_like(x)[:, 0, :])(encoder_o)
        initial_state = [initial_decoder_state, initial_attention_h]

        cell = DenseAnnotationAttention(cell=GRUCell(self.hidden_dim),
                                        units=self.hidden_dim,
                                        input_mode="concatenate",
                                        output_mode="cell_output")

        # TODO output_mode="concatenate", see TODO(3)/A
        decoder_o, decoder_h, decoder_c = RNN(cell=cell,
                                              return_sequences=True,
                                              return_state=True,
                                              name='decoder')(decoder_inputs_emb,
                                                              initial_state=initial_state,
                                                              constants=encoder_o)
        decoder_o = Dense(self.hidden_dim * 2, name='decoder_dense')(concatenate([decoder_o,
                                                                                  decoder_inputs_emb]))
        y_pred = Dense(self.num_decoder_tokens,
                       activation='softmax', name='summary_out')(decoder_o)

        model = Model([encoder_inputs, decoder_inputs], y_pred)
        return model


    def __k_largest_val_idx(self, a, k):
        """Returns top k largest values of a and their indices, ordered by
        decreasing value"""
        top_k = np.argpartition(a, -k)[-k:]
        return sorted(zip(a[top_k], top_k))[::-1]


    def build_inference_model(self):
        # build encoder
        x = self.summaryModel.get_layer('input_text').input
        h = self.summaryModel.get_layer('embedding_text')(x)
        x_enc = self.summaryModel.get_layer('encoder')(h)
        encoder_o = x_enc[0]
        initial_h_lstm = concatenate([x_enc[1], x_enc[2]])
        initial_c_lstm = concatenate([x_enc[3], x_enc[4]])
        initial_decoder_state = self.summaryModel.get_layer('decoder_state')(concatenate([initial_h_lstm,
                                                                                          initial_c_lstm]))
        initial_attention_h = Lambda(lambda x: K.zeros_like(x)[:, 0, :])(encoder_o)
        initial_state = [initial_decoder_state, initial_attention_h]

        encoder_model = Model(x, [encoder_o] + initial_state)

        # build decoder
        x_enc_new = Input(batch_shape=K.int_shape(encoder_o))
        y = self.summaryModel.get_layer('input_summary').input
        y_emb = self.summaryModel.get_layer('embedding_summary')(y)

        decoder = self.summaryModel.get_layer('decoder')
        initial_state_new = [Input((size,)) for size in decoder.cell.state_size]
        # initial_state_new = Input((size,))
        h1_and_state_new = decoder(y_emb, initial_state=initial_state_new, constants=x_enc_new)

        h1_new = h1_and_state_new[0]
        updated_state = h1_and_state_new[1:]
        h2_new = self.summaryModel.get_layer('decoder_dense')(concatenate([h1_new, y_emb]))
        y_pred_new = self.summaryModel.get_layer('summary_out')(h2_new)

        decoder_model = Model([y, x_enc_new] + initial_state_new,
                              [y_pred_new] + updated_state)

        self.encoder_model = encoder_model
        self.decoder_model = decoder_model


    def inference_beamsearch(self, input_text,
                             search_width=5,
                             branch_factor=None,
                             t_max=None):
        """Perform beam search to approximately find the translated sentence that
        maximises the conditional probability given the input sequence.

        Returns the completed sentences (reached end-token) in order of decreasing
        score (the first is most probable) followed by incomplete sentences in order
        of decreasing score - as well as the score for the respective sentence.

        References:
            [1] "Sequence to sequence learning with neural networks"
            (https://arxiv.org/pdf/1409.3215.pdf)
        """
        if branch_factor is None:
            branch_factor = search_width
        elif branch_factor > search_width:
            raise ValueError("branch_factor must be smaller than search_width")
        elif branch_factor < 2:
            raise ValueError("branch_factor must be >= 2")

        # initialisation of search
        t = 0
        y_0 = np.array(self.target_tokenizer.texts_to_sequences([self.start_token]))[0]
        end_idx = self.target_tokenizer.word_index[self.end_token]

        # run input encoding once
        x_ = np.array(self.input_tokenizer.texts_to_sequences([input_text]))
        encoder_output = self.encoder_model.predict(x_)
        x_enc_ = encoder_output[0]
        state_t = encoder_output[1:]
        # repeat to a batch of <search_width> samples
        x_enc_ = np.repeat(x_enc_, search_width, axis=0)

        if t_max is None:
            t_max = x_.shape[-1] * 2

        # A "search beam" is represented as the tuple:
        #   (score, outputs, state)
        # where:
        #   score: the average log likelihood of the output tokens
        #   outputs: the history of output tokens up to time t, [y_0, ..., y_t]
        #   state: the most recent state of the decoder_rnn for this beam

        # A list of the <search_width> number of beams with highest score is
        # maintained through out the search. Initially there is only one beam.
        incomplete_beams = [(0., [y_0], [s[0] for s in state_t])]
        # All beams that reached the end-token are kept separately.
        complete_beams = []

        while len(complete_beams) < search_width and t < t_max:
            t += 1
            # create a batch of inputs representing the incomplete_beams
            y_tm1 = np.vstack([beam[1][-1] for beam in incomplete_beams])
            state_tm1 = [
                np.vstack([beam[2][i] for beam in incomplete_beams])
                for i in range(len(state_t))
            ]

            # inference next tokes for every incomplete beam
            batch_size = len(incomplete_beams)
            decoder_output = self.decoder_model.predict(
                [y_tm1, x_enc_[:batch_size]] + state_tm1)

            y_pred_ = decoder_output[0]
            state_t = decoder_output[1:]
            # from each previous beam create new candidate beams and save the once
            # with highest score for next iteration.
            beams_updated = []
            for i, beam in enumerate(incomplete_beams):
                l = len(beam[1]) - 1  # don't count 'start' token
                for proba, idx in self.__k_largest_val_idx(y_pred_[i, 0], branch_factor):
                    idx += 1
                    new_score = (beam[0] * l + np.log(proba)) / (l + 1)
                    not_full = len(beams_updated) < search_width
                    ended = idx == end_idx
                    if not_full or ended or new_score > beams_updated[0][0]:
                        # create new successor beam with next token=idx
                        beam_new = (new_score,
                                    beam[1] + [np.array([idx])],
                                    [s[i] for s in state_t])
                        if ended:
                            complete_beams.append(beam_new)
                        elif not_full:
                            heapq.heappush(beams_updated, beam_new)
                        else:
                            heapq.heapreplace(beams_updated, beam_new)
                    else:
                        # if score is not among to candidates we abort search
                        # for this ancestor beam (next token processed in order of
                        # decreasing likelihood)
                        break
            # faster to process beams in order of decreasing score next iteration,
            # due to break above
            incomplete_beams = sorted(beams_updated, reverse=True)

        # want to return in order of decreasing score
        complete_beams = sorted(complete_beams, reverse=True)

        output_texts = []
        scores = []
        for beam in complete_beams + incomplete_beams:
            output_texts.append(self.target_tokenizer.sequences_to_texts(
                np.concatenate(beam[1])[None, :])[0])
            scores.append(beam[0])

        return output_texts, scores

    def inference_greedy(self, input_text, t_max=None):
        """Takes the most probable next token at each time step until the end-token
        is predicted or t_max reached.
        """
        t = 0
        y_t = np.array(self.target_tokenizer.texts_to_sequences([self.start_token]))
        y_0_to_t = [y_t]
        x_ = np.array(self.input_tokenizer.texts_to_sequences([input_text]))
        encoder_output = self.encoder_model.predict(x_)
        x_enc_ = encoder_output[0]
        state_t = encoder_output[1:]
        if t_max is None:
            t_max = x_.shape[-1] * 2
        end_idx = self.target_tokenizer.word_index[self.end_token]
        score = 0  # track the cumulative log likelihood
        while y_t[0, 0] != end_idx and t < t_max:
            t += 1
            decoder_output = self.decoder_model.predict([y_t, x_enc_] + state_t)
            y_pred_ = decoder_output[0]
            state_t = decoder_output[1:]
            y_t = np.argmax(y_pred_, axis=-1) + 1
            score += np.log(y_pred_[0, 0, y_t[0, 0]])
            y_0_to_t.append(y_t)
        y_ = np.hstack(y_0_to_t)
        output_text = self.target_tokenizer.sequences_to_texts(y_)[0]
        # length normalised score, skipping start token
        score = score / (len(y_0_to_t) - 1)

        return output_text, score


if __name__ == '__main__':
    model = seq2seq_attention(num_encoder_tokens=100, embedding_dim=64,
                              hidden_dim=128, num_decoder_tokens=50)
    summaryModel = model.get_model()
    summaryModel.compile(optimizer='Adam', loss='categorical_crossentropy')
