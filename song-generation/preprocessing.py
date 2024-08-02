## pip installs
## pip3 install pretty_midi

import pretty_midi
from random import randint
import torch
from torch.nn import functional
import os
import math 
from sklearn import utils



SEQUENCE_LENGTH = 8 # length that is to be used with for creating batches of training data.  # could be a hyper parameter.
START_PITCH = 21
END_PITCH = 108
NUM_PITCHES = 108 - 21 + 1 + 1 # +1 for clear pitch 0
BATCH_SIZE = 16
TRAIN_TEST_SPLIT = 0.8

# TESTING_MIDI_PATH = os.path.join("data","mozart-sonata.mid")
TESTING_MIDI_PATH = os.path.join("data","EtudeChopin.mid")
DATASET_PATH = None


print(TESTING_MIDI_PATH)

class Preprocessing:
  train_x = []
  train_y = []


  def convertFromMidiDataset(num_songs):
    ## returns  array of type MadeNote from midi file
    ## MadeNote is tuple with pitch: int and duration: float
    song_count = 0

    madeNoteArray = []
    for dir in os.listdir(DATASET_PATH):
      current_folder_path = os.path.join(DATASET_PATH,dir)
      if os.path.isdir(current_folder_path):
        if song_count > num_songs:
          break
        print(f"Loading {dir}.")
        for midi_path in os.listdir(current_folder_path):
          mz = pretty_midi.PrettyMIDI(os.path.join(current_folder_path,midi_path))
          piano_track = mz.instruments[0]
          sorted_track = sorted(piano_track.notes, key= lambda note: note.start)
          # print(sorted_track[:10])
          for note in sorted_track:
              newNote = (note.pitch,note.end - note.start)
              madeNoteArray.append(newNote)
          for _ in range(SEQUENCE_LENGTH - 1):
            madeNoteArray.append((0,0))
          song_count += 1



    return madeNoteArray
  def convertFromMidiPath(self,midi_path):
    ## returns  array of type MadeNote from midi file
    ## MadeNote is tuple with pitch: int and duration: float
    mz = pretty_midi.PrettyMIDI(midi_path)
    for instrument in mz.instruments:
      print(instrument)
    piano_track = mz.instruments[0]
    sorted_track = sorted(piano_track.notes, key= lambda note: note.start)
    # print(sorted_track[:10])
    madeNoteArray = []
    for note in sorted_track:
        newNote = (note.pitch,note.end - note.start)
        madeNoteArray.append(newNote)
    return madeNoteArray



  def createTraining(self,noteArray):
    ## extracts pitches and time from note array.


    X = []
    Y = []

    for i in range(len(noteArray)):
      if i + 1 == len(noteArray):
        pass

      new_x = []

      for j in range(SEQUENCE_LENGTH):
        backtrack_index = i - SEQUENCE_LENGTH + j
        if backtrack_index >= 0:  
          new_x.append(noteArray[backtrack_index])
        else:
          new_x.append((0,0))

      new_y = noteArray[i]
    
      X.append(new_x)
      Y.append(new_y)
    

    return torch.tensor(X),torch.tensor(Y)





  def randomSample(X,Y):
    i = randint(0,len(X))
    print(f"{X[i]} sequence \n is followed by \n {Y[i]}")


  def transformIndexing(self,X,Y):
    # pitch embedding
    assert len(X) != 0 and len(Y) != 0

    print("Training data shape:: ", X.shape)
    pitches_X = X[:,:,0]
    time_train_x = X[:,:,1]
    pitches_Y = Y[:,0]
    time_train_y = Y[:,1]

    assert (pitches_X.size()[0] == pitches_Y.size()[0])
     #making pitches ready for indexing for embedding
    for i in range(len(pitches_X)):
      for j in range(SEQUENCE_LENGTH):
        current_pitch_x = pitches_X[i,j]
        pitches_X[i,j] = current_pitch_x - 20 if current_pitch_x != 0 else 0
      current_pitch_y =  pitches_Y[i]
      pitches_Y[i] = current_pitch_y - 20 if current_pitch_y != 0 else 0
    return pitches_X, pitches_Y





  
  def shufflePitchesandSplit(self,X,Y):
    assert len(X) != 0 and len(Y) != 0
    
    shuffled_pitches_X,shuffled_pitches_Y = utils.shuffle(X,Y)

    split_index = math.floor(len(X) * TRAIN_TEST_SPLIT)

    ## splitting training and testing data.
    train_x = shuffled_pitches_X[:split_index]
    train_y = shuffled_pitches_Y[:split_index]
    test_x = shuffled_pitches_X[split_index:]
    test_y = shuffled_pitches_Y[split_index:]
    return train_x, train_y,test_x, test_y



  def binaryVectorization(self,data):
    # takes data of shape (n,SEQUENCE_LENGTH) 
    # and makes binary vectorized representaiton of shape (0,8,)
    one_hot_encoded = functional.one_hot(data.long(),num_classes = NUM_PITCHES)
    print("One hot encoded vector has dimensions: ", one_hot_encoded.shape)
    return one_hot_encoded

  def setData(self,num_songs,midi_path=None):
    if midi_path is None:
      noteArray = self.convertFromMidiDataset(DATASET_PATH,num_songs)
    else:
      noteArray = self.convertFromMidiPath(midi_path) # MIDI path used for dev
    # Make these funcitonal
    X,Y = self.createTraining(noteArray)
    X,Y = self.transformIndexing(X,Y)

    # to be accessed from getyData
    self.pitches_X = self.binaryVectorization(X)
    self.pitches_Y = self.binaryVectorization(Y)


  def getData(self):
    print(self.pitches_X.shape)
    print(self.pitches_Y.shape)
    # setData must be called before getData.
    # self.createTraining(noteArray)
    # self.transformIndexing()
    train_x, train_y, test_x, test_y = self.shufflePitchesandSplit(self.pitches_X, self.pitches_Y)
    return train_x, train_y, test_x, test_y
