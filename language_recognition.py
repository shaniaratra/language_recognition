import logging
import logging.config

from preprocessing import Preprocessing
from model import Model
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np

logging.config.fileConfig("logging.conf")
logger = logging.getLogger(__name__)


def main() -> None:
    """Main for the language recognition NN."""
    preprocess = Preprocessing("dataset.csv")
    data = preprocess.process()
    
    mapping = {language: index for index, language in enumerate(data["y_train"])}

    # model = Model(data)
    
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(data["x_train"])
    data["x_train"] = tokenizer.texts_to_sequences(data["x_train"])
    max_seq_length = max(len(seq) for seq in data["x_train"])
    data["x_train"] = tf.keras.preprocessing.sequence.pad_sequences(data["x_train"], maxlen=max_seq_length, padding='post')
        
    model = load_model('language_recognizer_model.keras')
    
    predictions = model.predict(data['x_train'], verbose=0)
    language_indexes = [np.argmax(prediction) for prediction in predictions]
    languages_detected = []
    for language_index in language_indexes:
        for language in mapping.keys():
            if mapping[language] == language_index:
                languages_detected.append(language)

    correct = 0
    incorrect = 0
    for language_detected, actual_value in zip(languages_detected, data['y_train']):
        if language_detected == actual_value:
            correct += 1
        else:
            incorrect += 1
                
                
    print(f'\n\nOn Train Data:')    
    print(f'Correctly identified: {correct}')
    print(f'Incorrectly identified: {incorrect}')
    error = incorrect/len(data['y_train'])
    print(f'Error: {error}')
    
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(data["x_test"])
    data["x_test"] = tokenizer.texts_to_sequences(data["x_test"])
    max_seq_length = max(len(seq) for seq in data["x_test"])
    data["x_test"] = tf.keras.preprocessing.sequence.pad_sequences(data["x_test"], maxlen=max_seq_length, padding='post')
        
    predictions = model.predict(data['x_test'], verbose=0)
    language_indexes = [np.argmax(prediction) for prediction in predictions]
    languages_detected = []
    for language_index in language_indexes:
        for language in mapping.keys():
            if mapping[language] == language_index:
                languages_detected.append(language)
    
    correct = 0
    incorrect = 0
    for language_detected, actual_value in zip(languages_detected, data['y_test']):
        if language_detected == actual_value:
            correct += 1
        else:
            incorrect += 1
    
    print(f'\n\nOn Test Data:')    
    print(f'Correctly identified: {correct}')
    print(f'Incorrectly identified: {incorrect}')
    error = incorrect/len(data['y_test'])
    print(f'Error: {error}')
            


if __name__ == "__main__":
    main()
