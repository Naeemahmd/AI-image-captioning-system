import numpy as np
from preprocessing import TokenizerWrapper
from preprocessing import (load_dataset_path, load_captions, get_image_paths_and_captions, extract_features, 
                           split_to_train_test, create_sequences)

from models import TransformerCaptioningModel
from evaluate import Evaluator

# load images and labelled captions
data_path = "archive/"
images_dir, captions_file = load_dataset_path(data_path)
captions = load_captions(captions_file)
image_paths, captions_list = get_image_paths_and_captions(captions, images_dir)

# Tokenize captions
all_captions = [caption for captions_list in captions.values() for caption in captions_list]
tokenizer = TokenizerWrapper(num_words=5000)
tokenizer.fit_on_texts(all_captions)
#extract features
image_features = extract_features(image_paths)

# Create train and validation sets
train_captions, val_captions = split_to_train_test(captions, test_size=0.1,)
vocab_size = tokenizer.get_vocab_size()
max_length = tokenizer.get_max_length()

X1train, X2train, ytrain = create_sequences(tokenizer, max_length, train_captions, image_features, vocab_size,image_paths)
X1val, X2val, yval = create_sequences(tokenizer, max_length, val_captions, image_features, vocab_size,image_paths)

#Build Model
embedding_matrix = np.zeros((vocab_size, 100))  # Assuming 100-dimensional GloVe embeddings
model_builder = TransformerCaptioningModel(vocab_size, max_length, embedding_matrix)
model = model_builder.build_model()

#Train Model
history = model.fit([X1train, X2train], ytrain, epochs=20, validation_data=([X1val, X2val], yval), verbose=2)

#Evaluate Model
evaluator = Evaluator(model, tokenizer, max_length)
bleu_scores = evaluator.evaluate_model(val_captions, image_features)

print("Done")