"""Preprocessing and cleaning language dataset."""

import logging
import logging.config
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models


logging.config.fileConfig("logging.conf")
logger = logging.getLogger(__name__)


class Model(tf.keras.Model):
    """Training and Testing a CNN."""

    def __init__(self, data):
        """Initialize Model Parameters."""
        super(Model, self).__init__()
        self.data = data
        
        tokenizer = tf.keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(self.data["x_train"])
        self.data["x_train"] = tokenizer.texts_to_sequences(self.data["x_train"])

        max_seq_length = max(len(seq) for seq in self.data["x_train"])
        self.data["x_train"] = tf.keras.preprocessing.sequence.pad_sequences(self.data["x_train"], maxlen=max_seq_length, padding='post')        
        mapping = {language: index for index, language in enumerate(self.data["y_train"])}
        languages = tf.keras.utils.to_categorical([mapping[language] for language in self.data['y_train']])
        
        self.model = models.Sequential(
            [
                layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=35, input_length=max_seq_length),
                layers.Bidirectional(layers.LSTM(70)), # Arbitrarily 70 for now
                layers.Dropout(0.3),
                layers.Dense(len(languages[0]), activation="softmax"),
            ]
        ) 
        
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')
        self.model.fit(data['x_train'], languages, epochs=5, batch_size=20)
        self.model.save('language_recognizer_model.keras')
        
        predictions = self.model.predict(data['x_train'], verbose=0)
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