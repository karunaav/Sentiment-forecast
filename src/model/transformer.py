import tensorflow as tf
from tensorflow.keras import layers, models


def build_transformer(input_dim, window=10, num_heads=4, ff_dim=64, dropout=0.1):
    """
    Build a simple encoder-only Transformer for time series regression.
    input_dim: number of features
    window: number of time steps to look back (context length)
    """
    inputs = layers.Input(shape=(input_dim,))
    
    # Expand input for attention
    x = layers.RepeatVector(window)(inputs)
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    # Transformer encoder block
    attn_out = layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_dim)(x, x)
    attn_out = layers.Dropout(dropout)(attn_out)
    x = layers.Add()([x, attn_out])
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff = layers.Dense(input_dim)(ff)
    x = layers.Add()([x, ff])
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1)(x)

    model = models.Model(inputs, outputs, name="transformer_encoder")
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model
