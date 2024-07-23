from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import tensorflow as tf

class Evaluator:
    def __init__(self, model, tokenizer, max_length):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def generate_desc(self, photo):
        in_text = 'startseq'
        for _ in range(self.max_length):
            sequence = self.tokenizer.texts_to_sequences(in_text)
            sequence = self.tokenizer.pad_sequences(sequence)
            yhat = self.model.predict([photo, sequence], verbose=0)
            yhat = tf.argmax(yhat, axis=-1).numpy()
            word = self.tokenizer.index_word.get(yhat[-1], None)
            if word is None:
                break
            in_text += ' ' + word
            if word == 'endseq':
                break
        return in_text

    def evaluate_model(self, descriptions, photos):
        actual, predicted = list(), list()
        for key, desc_list in descriptions.items():
            yhat = self.generate_desc(photos[key])
            references = [d.split() for d in desc_list]
            actual.append(references)
            predicted.append(yhat.split())
        smoothie = SmoothingFunction().method4
        bleu1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0), smoothing_function=smoothie)
        bleu2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
        bleu3 = corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0), smoothing_function=smoothie)
        bleu4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
        print(f'BLEU-1: {bleu1}')
        print(f'BLEU-2: {bleu2}')
        print(f'BLEU-3: {bleu3}')
        print(f'BLEU-4: {bleu4}')
        return bleu1, bleu2, bleu3, bleu4
