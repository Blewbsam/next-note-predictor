
from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo, second2tick
import os

SAVINGPATH =  os.path.join("generated-songs")

class PostProcessor:


    def convertNotesToMidiAndSave(self,notes):

        mid = MidiFile(type=0)
        track = MidiTrack()
        mid.tracks.append(track)

        track.append(MetaMessage('key_signature', key='Cm'))
        track.append(MetaMessage('set_tempo', tempo=bpm2tempo(90)))
        track.append(MetaMessage('time_signature', numerator=4, denominator=4))

    

        for i in range(len(notes)):
            if i == 0:
                track.append(Message('note_on', channel=2, note=notes[i], velocity=64, time=200))
            else: 
                track.append(Message('note_on', channel=2, note=notes[i], velocity=64, time=0))

            track.append(Message('note_off', channel=2, note=notes[i], velocity=64, time=200))

        track.append(MetaMessage('end_of_track'))
        mid.save(filename="gen40.mid")
        print(":-)")

