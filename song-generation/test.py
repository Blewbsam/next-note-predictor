import pretty_midi
import os

def convertFromMidiPath(midi_path):
    ## returns  array of type MadeNote from midi file
    ## MadeNote is tuple with pitch: int and duration: float
    mz = pretty_midi.PrettyMIDI(midi_path)
    for instrument in mz.instruments:
        print(instrument)
    piano_track = mz.instruments[0]
    sorted_track = sorted(piano_track.notes, key= lambda note: note.start)
    # print(sorted_track[:10])
    count = 0
    for note in sorted_track:
        print(note)
        count += 1
        if (count > 20):
            break



chopin_path = os.path.join("song-generation","data","EtudeChopin.mid")
convertFromMidiPath(chopin_path)

# for path in os.listdir("data"):
    # print(path)
