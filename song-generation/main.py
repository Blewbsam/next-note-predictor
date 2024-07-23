
from model import Model

def main():
    engine = Model()
    engine.setUpParameters()
    engine.trainModel(epochs=40)


main()