import tensorflow as tf
from tensorflow.keras.layers import Embedding, Add, TimeDistributed, Dense, Input
from tensorflow.keras.models import Model

class CNNExtractor:
    def __init__(self, input_shape=(2048,)):
        self.input_shape = input_shape

    def build_model(self):
        cnn_input = Input(shape=self.input_shape)
        cnn_layer = Dense(256, activation='relu')(cnn_input)
        return Model(inputs=cnn_input, outputs=cnn_layer)

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim)]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TransformerCaptioningModel:
    def __init__(self, vocab_size, max_length, embedding_matrix):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_matrix = embedding_matrix

    def build_model(self):
        cnn_extractor = CNNExtractor().build_model()
        
        text_input = Input(shape=(self.max_length,))
        embedding = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_matrix.shape[1], 
                              weights=[self.embedding_matrix], trainable=False)(text_input)
        transformer_block = TransformerBlock(embed_dim=self.embedding_matrix.shape[1], num_heads=8, ff_dim=256, rate=0.2)
        transformer_output = transformer_block(embedding, training=True)
        
        # Align the dimensions of the outputs from CNN and Transformer
        cnn_transformed = Dense(self.embedding_matrix.shape[1])(cnn_extractor.output)
        
        combined_input = Add()([cnn_transformed, transformer_output])
        combined_output = TimeDistributed(Dense(256))(combined_input)
        combined_output = TimeDistributed(Dense(self.vocab_size, activation='softmax'))(combined_output)

        model = Model(inputs=[cnn_extractor.input, text_input], outputs=combined_output)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        return model
