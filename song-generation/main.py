from model import Model
from preprocessing import Preprocessing
from postprocessing import PostProcessor
import os

CHOPIN_DATA = os.path.join("song-generation","data","EtudeChopin.mid")

def main():
    print(":)")
    engine = Model()
    preprocessor = Preprocessing()
    preprocessor.setData(num_songs=300,midi_path=None)

    engine.setUpParameters()
    engine.countParameters()
    engine.trainModel(preprocessor,epochs=40,verbose=True)
    generatedNotes = engine.generateNotes(400)

    postProcessor = PostProcessor()
    postProcessor.convertNotesToMidiAndSave(generatedNotes)




if __name__ == "__main__":
    main()   