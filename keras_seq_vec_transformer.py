'''
Based on:
https://www.analyticsvidhya.com/blog/2019/06/understanding-transformers-nlp-state-of-the-art-models/
https://arxiv.org/abs/1706.03762
'''

import numpy as np

import keras
import keras.backend as K
from keras_layer_normalization import LayerNormalization

class MultiHeadAttention(keras.layers.Layer):
    
    def __init__(self, n_heads, d_key, d_value, **kwargs):
        self.n_heads = n_heads
        self.d_key = d_key
        self.sqrt_d_key = np.sqrt(self.d_key)
        self.d_value = d_value
        self.d_output = n_heads * d_value
        super(MultiHeadAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        
        # input_shape: (batch_size, length, d_input)
        _, _, self.d_input = input_shape
        
        # Wq, Wk: (n_heads, d_input, d_key)
        self.Wq = self.add_weight(name = 'Wq', shape = (self.n_heads, self.d_input, self.d_key), \
                initializer = 'glorot_uniform', trainable = True)
        self.Wk = self.add_weight(name = 'Wk', shape = (self.n_heads, self.d_input, self.d_key), \
                initializer = 'glorot_uniform', trainable = True)
        
        # Wv: (n_heads, d_input, d_value)
        self.Wv = self.add_weight(name = 'Wv', shape = (self.n_heads, self.d_input, self.d_value), \
                initializer = 'glorot_uniform', trainable = True)

        super(MultiHeadAttention, self).build(input_shape)

    def call(self, X):
        
        # X: (batch_size, length, d_input)
        _, length, d_input = K.int_shape(X)
        assert d_input == self.d_input
        
        QX = K.tanh(K.dot(X, self.Wq)) # (batch_size, length, n_heads, d_key)
        # (batch_size * n_heads, length, d_key)
        QX = K.reshape(K.permute_dimensions(QX, (0, 2, 1, 3)), (-1, length, self.d_key)) 
        
        KX = K.tanh(K.dot(X, self.Wk)) # (batch_size, length, n_heads, d_key)
        # (batch_size * n_heads, length, d_key)
        KX = K.reshape(K.permute_dimensions(KX, (0, 2, 1, 3)), (-1, length, self.d_key)) 

        VX = K.relu(K.dot(X, self.Wv)) # (batch_size, length, n_heads, d_value)
        # (batch_size * n_heads, length, d_value)
        VX = K.reshape(K.permute_dimensions(VX, (0, 2, 1, 3)), (-1, length, self.d_value)) 

        # (batch_size * n_heads, length, length)
        Z = K.softmax(K.batch_dot(QX, K.permute_dimensions(KX, (0, 2, 1))) / self.sqrt_d_key)
        Y = K.batch_dot(Z, VX) # (batch_size * n_heads, length, d_value)
        # (batch_size, length, n_heads, d_value)
        Y = K.permute_dimensions(K.reshape(Y, (-1, self.n_heads, length, self.d_value)), (0, 2, 1, 3))
        # (batch_size, length, n_heads * d_value)
        return K.reshape(Y, (-1, length, self.d_output))

    def compute_output_shape(self, input_shape):
        # input_shape: (batch_size, length, d_input)
        batch_size, length, _ = input_shape
        return (batch_size, length, self.d_output)

class TransformerBlock(keras.layers.Layer):
        
    def __init__(self, n_heads, d_seq, d_key, d_vec, dense_activation = 'relu', **kwargs):
        
        assert d_seq % n_heads == 0
        self.n_heads = n_heads
        self.d_seq = d_seq
        self.d_key = d_key
        self.d_vec = d_vec
        self.d_value = d_seq // n_heads
        
        name = kwargs.get('name', 'transformer')
        self.attention = MultiHeadAttention(self.n_heads, self.d_key, self.d_value, name = '%s-attention' % name)
        self.attention_norm = LayerNormalization(name = '%s-attention-norm' % name)
        self.seq_dense1 = keras.layers.Dense(self.d_seq, activation = dense_activation, name = '%s-seq-dense1' % name)
        self.seq_norm1 = LayerNormalization(name = '%s-seq-norm1' % name)
        self.vec_dense1 = keras.layers.Dense(self.d_vec, activation = dense_activation, name = '%s-vec-dense1' % name)
        self.seq_mean_dense = keras.layers.Dense(self.d_vec, activation = dense_activation, name = '%s-seq-mean-dense' % name)
        self.vec_norm1 = LayerNormalization(name = '%s-vec-norm1' % name)
        self.vec_dense2 = keras.layers.Dense(self.d_vec, activation = dense_activation, name = '%s-vec-dense2' % name)
        self.vec_norm2 = LayerNormalization(name = '%s-vec-norm2' % name)
        self.vec_seqing_dense = keras.layers.Dense(self.d_seq, activation = dense_activation, name = '%s-vec-seqing-dense' % name)
        self.seq_dense2 = keras.layers.Dense(self.d_seq, activation = dense_activation, name = '%s-seq-dense2' % name)
        self.seq_norm2 = LayerNormalization(name = '%s-seq-norm2' % name)
        self.seq_dense3 = keras.layers.Dense(self.d_seq, activation = dense_activation, name = '%s-seq-dense3' % name)
        self.seq_norm3 = LayerNormalization(name = '%s-seq-norm3' % name)
        
        self.layers_with_seq_input = [self.attention, self.attention_norm, self.seq_dense1, self.seq_norm1, self.seq_dense2, \
                self.seq_norm2, self.seq_dense3, self.seq_norm3]
        self.layers_with_vec_input = [self.vec_dense1, self.vec_norm1, self.vec_dense2, self.vec_norm2, self.vec_seqing_dense]
        self.all_layers = self.layers_with_seq_input + self.layers_with_vec_input + [self.seq_mean_dense]
        
        super(TransformerBlock, self).__init__(**kwargs)
        
    def build(self, input_shapes):
        
        seq_shape, vec_shape = input_shapes
        batch_size, length, d_seq = seq_shape
        batch_size2, d_vec = vec_shape
        assert batch_size == batch_size2
        assert d_seq == self.d_seq
        assert d_vec == self.d_vec
        
        for layer in self.layers_with_seq_input:
            layer.build(seq_shape)
            
        for layer in self.layers_with_vec_input:
            layer.build(vec_shape)
            
        self.seq_mean_dense.build((batch_size, d_seq))
                
        self._trainable_weights = [weight for layer in self.all_layers for weight in layer._trainable_weights]
        super(TransformerBlock, self).build([seq_shape, vec_shape])
        
    def compute_output_shape(self, input_shapes):
        return input_shapes

    def call(self, X):
        
        X_seq, X_vec = X
        
        X_seq = self.attention_norm(keras.layers.Add()([X_seq, self.attention(X_seq)]))
        X_seq = self.seq_norm1(keras.layers.Add()([X_seq, self.seq_dense1(X_seq)]))
        
        # (batch_size, length, d_seq) --> (batch_size, d_seq)
        X_seq_mean = K.mean(X_seq, axis = 1)
        X_vec = self.vec_norm1(keras.layers.Add()([X_vec, self.vec_dense1(X_vec), self.seq_mean_dense(X_seq_mean)]))
        X_vec = self.vec_norm2(keras.layers.Add()([X_vec, self.vec_dense2(X_vec)]))
        
        # (batch_size, d_vec) --> (batch_size, 1, d_seq)
        X_vec_seqed = K.expand_dims(self.vec_seqing_dense(X_vec), axis = 1)
        X_seq = self.seq_norm2(keras.layers.Add()([X_seq, self.seq_dense2(X_seq), X_vec_seqed]))
        X_seq = self.seq_norm3(keras.layers.Add()([X_seq, self.seq_dense3(X_seq)]))
        
        return [X_seq, X_vec]
    
def get_sinusoidal_embedding(d_pos_vec, n_position):
    '''
    Taken from: https://stackoverflow.com/questions/46452020/sinusoidal-embedding-attention-is-all-you-need
    '''
    position_enc = np.array([
        [pos / np.power(10000, 2*i/d_pos_vec) for i in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])
    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return position_enc
    
class SeqInputEmbedding(keras.layers.Layer):
    
    def __init__(self, vocab_size, d_positional_embedding, **kwargs):
        self.vocab_size = vocab_size
        self.d_positional_embedding = d_positional_embedding
        self.d_total = self.vocab_size + self.d_positional_embedding
        super(SeqInputEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: (batch_size, length)
        _, self.length = input_shape
        # (length, d_positional_embedding)
        self.position_embeddings = K.constant(get_sinusoidal_embedding(self.d_positional_embedding, self.length))
        super(SeqInputEmbedding, self).build(input_shape)

    def call(self, X):
        
        # X is the input tokens, given as integers of shape (batch_size, length)
        _, length = K.int_shape(X)
        assert length == self.length
        batch_size = K.shape(X)[0]
        
        # (batch_size, length, vocab_size)
        token_embedding = K.one_hot(X, self.vocab_size)
        # (batch_size, length, d_positional_embedding)
        positional_embedding = K.tile(K.reshape(self.position_embeddings, (1, length, self.d_positional_embedding)), \
                (batch_size, 1, 1))
        # (batch_size, length, d_total)
        return K.concatenate([token_embedding, positional_embedding])

    def compute_output_shape(self, input_shape):
        # input_shape: (batch_size, length)
        batch_size, length = input_shape
        return (batch_size, length, self.d_total)    
    
class TransformerAutoEncoder(keras.layers.Layer):
    
    '''
    # TODO set d_vec = 0 by default to mean no input/output vector. 
    '''
    
    def __init__(self, vocab_size, d_vec, n_transformers = 6, n_heads = 8, d_hidden_seq = 512, d_key = 64, \
            d_hidden_vec = 512, d_positional_embedding = 16, hidden_activation = 'relu', output_vec_activation = None, \
            output_vec = True, **kwargs):

        self.vocab_size = vocab_size
        self.d_vec = d_vec
        self.n_transformers = n_transformers
        self.n_heads = n_heads
        self.d_hidden_seq = d_hidden_seq
        self.d_key = d_key
        self.d_hidden_vec = d_hidden_vec
        self.d_positional_embedding = d_positional_embedding
        self.hidden_activation = hidden_activation
        self.output_vec_activation = output_vec_activation
        self.output_vec = output_vec
        
        self.seq_input_embedding = SeqInputEmbedding(vocab_size = vocab_size, \
                d_positional_embedding = d_positional_embedding, name = 'input-embedding')
        self.input_seq_dense = keras.layers.Dense(d_hidden_seq, activation = hidden_activation, name = 'input-seq-dense')
        self.input_vec_dense = keras.layers.Dense(d_hidden_vec, activation = hidden_activation, name = 'input-vec-dense')
        self.transformer_blocks = [TransformerBlock(n_heads = n_heads, d_seq = d_hidden_seq, d_key = d_key, \
                d_vec = d_hidden_vec, dense_activation = hidden_activation, name = 'transformer-%d' % (i + 1)) for i in \
                range(n_transformers)]
        self.output_seq_dense = keras.layers.Dense(vocab_size, activation = 'softmax', name = 'output-seq-dense')
        
        self.all_layers = [self.seq_input_embedding, self.input_seq_dense, self.input_vec_dense, self.output_seq_dense] + \
                self.transformer_blocks
        
        if self.output_vec:
            self.output_vec_dense = keras.layers.Dense(d_vec, activation = output_vec_activation, name = 'output-vec-dense')
            self.all_layers.append(self.output_vec_dense)
        
        super(TransformerAutoEncoder, self).__init__(**kwargs)
        
    def build(self, input_shapes):
        
        seq_input_shape, vec_shape = input_shapes
        
        self.seq_input_embedding.build(seq_input_shape)
        embedded_seq_shape = self.seq_input_embedding.compute_output_shape(seq_input_shape)
        self.input_seq_dense.build(embedded_seq_shape)
        hidden_seq_shape = self.input_seq_dense.compute_output_shape(embedded_seq_shape)
        self.input_vec_dense.build(vec_shape)
        hidden_vec_shape = self.input_vec_dense.compute_output_shape(vec_shape)
        
        for transformer_block in self.transformer_blocks:
            transformer_block.build([hidden_seq_shape, hidden_vec_shape])
            
        self.output_seq_dense.build(hidden_seq_shape)
        
        if self.output_vec:
            self.output_vec_dense.build(hidden_vec_shape)
            
        self._trainable_weights = [weight for layer in self.all_layers for weight in layer._trainable_weights]
        super(TransformerAutoEncoder, self).build([seq_input_shape, vec_shape])
        
    def compute_output_shape(self, input_shapes):
        
        seq_input_shape, vec_shape = input_shapes
        batch_size, length = seq_input_shape
        seq_output_shape = (batch_size, length, self.vocab_size)
        
        if self.output_vec:
            return [seq_output_shape, vec_shape]
        else:
            return seq_output_shape

    def call(self, X):
        
        X_seq, X_vec = X
        
        X_seq = self.input_seq_dense(self.seq_input_embedding(X_seq))
        X_vec = self.input_vec_dense(X_vec)
  
        for transformer_block in self.transformer_blocks:
            X_seq, X_vec = transformer_block([X_seq, X_vec])
            
        X_seq = self.output_seq_dense(X_seq)
        
        if self.output_vec:
            X_vec = self.output_vec_dense(X_vec)
            return [X_seq, X_vec]
        else:
            return X_seq