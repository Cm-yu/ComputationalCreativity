# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 02:07:44 2020

@author: cheng
"""

import tensorflow as tf
import os
import random
from music21 import converter, instrument, note, chord, stream

import numpy as np
from numpy.random import seed

seed(1)
tf.random.set_seed(2)

def get_notes(tag):
    filepath = 'music_train/{}/'.format(tag)
    files = os.listdir(filepath)
    Notes= []
    for file in files:
        st = converter.parse(filepath+file)
        parts = instrument.partitionByInstrument(st)
        if parts:
            for part in parts:
                if 'Piano' in str(part):
                    notes = part.recurse()      
                    for element in notes:
                        if isinstance(element, note.Note):
                            Notes.append(str(element.pitch))
                        elif isinstance(element, chord.Chord):
                            Notes.append('.'.join(str(n) for n in element.normalOrder))                  
    with open('data/notes/{}'.format(tag), 'w') as f:
        f.write(str(Notes))       
    return Notes                                             
        
def network_model(inputs, num_pitch, weights_file=None):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(512,input_shape=(inputs.shape[1], inputs.shape[2]), return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.LSTM(512, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.LSTM(512))
    model.add(tf.keras.layers.Dense(256))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(num_pitch))
    model.add(tf.keras.layers.Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
 
    if weights_file is not None:
        model.load_weights(weights_file)
    return model

def prepare_sequences(notes, num_pitch):
    sequence_length = 100
    pitch = sorted(set(item for item in notes))
 
    note_to_int = dict((note, number) for number, note in enumerate(pitch))
 
    X = []
    y = []
 
    for i in range(0, len(notes) - sequence_length, 1):
        input_  = notes[i: i + sequence_length]
        output = notes[i + sequence_length]
        X.append([note_to_int[k] for k in input_])
        y.append(note_to_int[output])

    X = np.reshape(X, (len(X), sequence_length, 1))

    X = X / float(num_pitch)
    y = tf.keras.utils.to_categorical(y)
 
    return X, y
 
def train(tag):
    notes = get_notes(tag)
    num_pitch = len(set(notes))
    network_input, network_output = prepare_sequences(notes, num_pitch)
 
    model = network_model(network_input, num_pitch)

    filepath = "weights/" + tag + "/weights.{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]
    model.fit(network_input, network_output, epochs=100, batch_size=128, callbacks=callbacks_list)


def produce_notes(model, network_input, pitch, num_pitch):
    start = np.random.randint(0, len(network_input) - 1)
    int_to_note = dict((number, note) for number, note in enumerate(pitch))
 
    pattern = network_input[start]
 
    prediction_output = []
 
    for note_index in range(50):
        input_  = np.reshape(pattern, (1, len(pattern), 1))
        input_  = input_ / float(num_pitch)
        proba  = model.predict(input_, verbose=0)
 
        index = np.argmax(proba)
 
        pred  = int_to_note[index]
 
        prediction_output.append(pred)
        
        pattern = list(pattern)
        pattern.append([index/float(num_pitch)])
        pattern = pattern[1:len(pattern)]
        
        print(note_index)
    return prediction_output


def create_music(prediction,tag):
    offset = 0
    output_notes = []

    for data in prediction:
        if ('.' in data) or data.isdigit():
            notes_in_chord = data.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(data)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        offset += 0.5
 
    midi_stream = stream.Stream(output_notes)
 
    midi_stream.write('midi', fp='output_{}.mid'.format(tag))
    


def produce(tag):
    notes = get_notes(tag)
    pitch_names = sorted(set(item for item in notes))
    num_pitch = len(set(notes))
    network_input, network_output = prepare_sequences(notes, num_pitch)
    files = os.listdir("weights/{}/".format(tag))
    minloss = {}
    for i in files:
        if 'weights' in i:
            num = i[11:15]
            minloss[num] = i
    best_weights = minloss[min(minloss.keys())]
    model = network_model(network_input, num_pitch, "weights/{}/{}".format(tag,best_weights))
    prediction = produce_notes(model, network_input, pitch_names, num_pitch)
    create_music(prediction,tag)

#only need to run once to gain best weights
train('classical') 
 
tag = ['classical']
produce(random.choice(tag))
