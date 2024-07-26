## pip installs
## pip3 install pretty_midi

import pretty_midi
from random import randint
import torch
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
      
    self.X = torch.tensor(X)
    self.Y = torch.tensor(Y)

    print(len(X))
    print(len(Y))





  def randomSample(X,Y):
    i = randint(0,len(X))
    print(f"{X[i]} sequence \n is followed by \n {Y[i]}")


  def transformIndexing(self):
    # pitch embedding
    assert len(self.X) != 0 and len(self.Y) != 0

    print("Training data shape:: ", self.X.shape)
    self.pitches_X = self.X[:,:,0]
    time_train_x = self.X[:,:,1]
    self.pitches_Y = self.Y[:,0]
    time_train_y = self.Y[:,1]

    assert (self.pitches_X.size()[0] == self.pitches_Y.size()[0])

     #making pitches ready for indexing for embedding
    for i in range(len(self.pitches_X)):
      for j in range(SEQUENCE_LENGTH):
        current_pitch_x = self.pitches_X[i,j]
        self.pitches_X[i,j] = current_pitch_x - 20 if current_pitch_x != 0 else 0
      current_pitch_y =  self.pitches_Y[i]
      self.pitches_Y[i] = current_pitch_y - 20 if current_pitch_y != 0 else 0




  
  def shufflePitchesandSplit(self):
    assert len(self.pitches_X) != 0 and len(self.pitches_Y) != 0
    
    shuffled_pitches_X,shuffled_pitches_Y = utils.shuffle(self.pitches_X,self.pitches_Y)

    split_index = math.floor(len(self.pitches_X) * TRAIN_TEST_SPLIT)

    ## splitting training and testing data.
    self.train_x = shuffled_pitches_X[:split_index]
    self.train_y = shuffled_pitches_Y[:split_index]
    self.test_x = shuffled_pitches_X[split_index:]
    self.test_y = shuffled_pitches_Y[split_index:]



  def binaryVectorization(self,data):
    # takes data of shape (n,SEQUENCE_LENGTH) 
    # and makes binary vectorized representaiton of shape (0,8,)

    assert data.ndim == 2

    binary_pitches = torch.empty(0,SEQUENCE_LENGTH,89)
    for row in data:
      current_row = torch.empty(0,0,NUM_PITCHES)
      # current_row = []
      for note in row:
        curBinaryNote = torch.zeros((0,1,NUM_PITCHES))
        index = note.int().item()
        if (index != 0):
          curBinaryNote[:,:,index] = 1
        current_row = torch.cat((current_row,curBinaryNote),dim=1)
      binary_pitches = torch.cat((binary_pitches,current_row),dim=2)
    




  def setData(self,num_songs,midi_path=None):
    if midi_path is None:
      self.noteArray = self.convertFromMidiDataset(DATASET_PATH,num_songs)
    else:
      self.noteArray = self.convertFromMidiPath(midi_path) # MIDI path used for dev
    print("Got data")
    self.createTraining(self.noteArray)
    self.transformIndexing()

  def getData(self):
    # setData must be called before getData.
    # self.createTraining(self.noteArray)
    # self.transformIndexing()
    self.shufflePitchesandSplit()
    return self.train_x, self.train_y, self.test_x,self.test_y
