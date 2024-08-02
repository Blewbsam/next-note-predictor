from model import Model
from preprocessing import Preprocessing
import os

CHOPIN_DATA = os.path.join("song-generation","data","EtudeChopin.mid")

def main():
    print(":)")
    engine = Model()
    preprocessor = Preprocessing()
    preprocessor.setData(num_songs=10,midi_path=None)

    engine.setUpParameters()
    engine.countParameters()
    engine.trainModel(preprocessor,epochs=100,verbose=True)


if __name__ == "__main__":
    main()  