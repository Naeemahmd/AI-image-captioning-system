import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Add, LayerNormalization, MultiHeadAttention
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu
from PIL import Image


# Tokenize captions
def tokenize_captions(captions_list):
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(captions_list)
    sequences = tokenizer.texts_to_sequences(captions_list)
    max_length = max(len(seq) for seq in sequences)
    captions_padded = pad_sequences(sequences, maxlen=max_length, padding='post')
    return tokenizer, captions_padded, max_length


# Extract features using ResNet50
def extract_features(image_paths):
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    features = {}
    for img_path in image_paths:
        image = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = tf.keras.applications.resnet50.preprocess_input(image)
        feature = model.predict(image, verbose=0)
        img_id = os.path.basename(img_path).split('.')[0]
        features[img_id] = feature
    return features