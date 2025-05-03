import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

NUM_FEATURE = 8  # Replace with the correct value
SEQ_LENGTH = 512  # Replace with the correct value
EMBED_DIM = 512  # Replace with the correct value

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, embed_dim, trainable_embed=False, **kwargs):
        super().__init__(**kwargs)
        
        # Generate positional encoding matrix
        position_embedding_matrix = self.get_position_encoding(sequence_length, embed_dim)
        
        # Non-trainable Embedding layer to store positional encodings
        if trainable_embed == False:
            self.position_embedding_layer = layers.Embedding(
                input_dim=sequence_length, output_dim=embed_dim,
                weights=[position_embedding_matrix],
                trainable=False
            )
        else:
            self.position_embedding_layer = layers.Embedding(
                input_dim=sequence_length, output_dim=embed_dim,
                trainable=True
            )
             
    def get_position_encoding(self, seq_len, d, n=10000):
        P = np.zeros((seq_len, d))
        for k in range(seq_len):
            for i in np.arange(int(d/2)):
                denominator = np.power(n, 2*i/d)
                P[k, 2*i] = np.sin(k/denominator)
                P[k, 2*i+1] = np.cos(k/denominator)
        return P
 
    def call(self, inputs):        
        # Get position indices (0, 1, ..., sequence_length-1)
        sequence_length = tf.shape(inputs)[1]  # Ambil sequence_length dari inputs
        position_indices = tf.range(sequence_length)  # Shape: (sequence_length,)
        
        # Get positional encoding
        positional_encoding = self.position_embedding_layer(position_indices)  # Shape: (sequence_length, embed_dim)
        
        # Expand dimensions for broadcasting: (1, sequence_length, embed_dim)
        return positional_encoding[tf.newaxis, :, :]


class TransformerEncoderBlock(tf.keras.Model):
    def __init__(self, d_models, num_heads, num_layers=6, **kwargs):
        super().__init__(**kwargs)

        self.dense256 = layers.Dense(256, activation="relu")
        self.dense_transform = layers.Dense(d_models, activation="relu")
        self.num_layers = num_layers
        self.linear = layers.Dense(d_models, activation="relu")

        self.frame_positional_encoding = PositionalEncoding(
            sequence_length=NUM_FEATURE + 1, embed_dim=d_models, trainable_embed=True
        )
        self.patch_positional_encoding = PositionalEncoding(
            sequence_length=SEQ_LENGTH + 1, embed_dim=d_models, trainable_embed=True
        )

        # Define layers as lists
        self.attention_frame = [layers.MultiHeadAttention(num_heads=num_heads, key_dim=int(d_models/num_heads), dropout=0.2) 
                                for _ in range(num_layers)]
        self.attention_cross = [layers.MultiHeadAttention(num_heads=num_heads, key_dim=int(d_models/num_heads), dropout=0.2) 
                                for _ in range(num_layers)]
        self.layernorm_frame = [layers.LayerNormalization() for _ in range(num_layers)]
        self.layernorm_cross = [layers.LayerNormalization() for _ in range(num_layers)]
        self.dense_1 = [layers.Dense(d_models*2, activation="gelu") for _ in range(num_layers)]
        self.dense_2 = [layers.Dense(d_models) for _ in range(num_layers)]
        self.dropout = [layers.Dropout(0.2) for _ in range(num_layers)]

        # Learnable class tokens
        self.cls_token = self.add_weight(
            shape=(1, 1, d_models),
            initializer="random_normal",
            trainable=True,
            name="cls_token"
        )

        self.cls_token2 = self.add_weight(
            shape=(1, 1, d_models),
            initializer="random_normal",
            trainable=True,
            name="cls_token2"
        )

        # Final output layers
        self.out = layers.Dense(d_models, activation="gelu")
        self.out_drop = layers.Dropout(0.2)

    def call(self, inputs, training=False):
        """Transformer Encoder Forward Pass"""
        inputs, frame = inputs

        # Add class token
        batch_size = tf.shape(inputs)[0]
        inputs = self.dense_transform(self.dense256(inputs))
        cls_tokens = tf.broadcast_to(self.cls_token, [batch_size, 1, tf.shape(self.cls_token)[-1]])
        cls_tokens2 = tf.broadcast_to(self.cls_token2, [batch_size, 1, tf.shape(self.cls_token2)[-1]])

        # Process frame
        frame = self.linear(frame)
        frame = tf.concat([cls_tokens2, frame], axis=1)
        Zt = layers.Add()([frame, self.frame_positional_encoding(frame)])

        # Process inputs
        inputs = tf.concat([cls_tokens, inputs], axis=1)
        Zs = layers.Add()([inputs, self.patch_positional_encoding(inputs)])

        # Iterate over pre-defined layers (avoid Graph Mode error)
        for i in range(self.num_layers):
            Zt = self.layernorm_frame[i](Zt + self.attention_frame[i](query=Zt, value=Zt, key=Zt, training=training))
            Zts = tf.concat([Zs, Zt], axis=1)
            Zs = self.layernorm_cross[i](Zs + self.attention_cross[i](query=Zs, value=Zts, key=Zts, training=training))

            inputs = self.dense_1[i](Zs)
            inputs = self.dropout[i](inputs, training=training)
            inputs = self.dense_2[i](inputs)
            Zs += inputs

            inputsf = self.dense_1[i](Zt)
            inputsf = self.dropout[i](inputsf, training=training)
            inputsf = self.dense_2[i](inputsf)
            Zt += inputsf

        # Output
        Zt = Zt[:, 0, :]
        Zs = Zs[:, 0, :]
        out = tf.concat([Zs, Zt], axis=-1)
        out = self.out(out)
        out = self.out_drop(out)
        return out


    def create_look_ahead_mask(self, size):
        return tf.linalg.band_part(tf.ones((size, size)), -1, 0)


class Classifier(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.densel = layers.Dense(EMBED_DIM)
        self.out = layers.Dense(3, activation="softmax")

    def call(self, inputs):
        x = self.densel(inputs)
        x = self.out(x)
        return x