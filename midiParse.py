import pretty_midi

def separate_guitar_tracks(midi_path, output_chords, output_solo):
    # Load MIDI file
    midi_data = pretty_midi.PrettyMIDI(midi_path)

    # Create new MIDI files for chords and solo
    chords_midi = pretty_midi.PrettyMIDI()
    solo_midi = pretty_midi.PrettyMIDI()

    chords_instrument_electric = pretty_midi.Instrument(program=28)
    chords_instrument_acoustic = pretty_midi.Instrument(program=26)
    solo_instrument_electric = pretty_midi.Instrument(program=28)
    solo_instrument_acoustic = pretty_midi.Instrument(program=26)

    for instrument in set(midi_data.instruments):
        # Skip non-guitar instruments
        chords_instrument = chords_instrument_acoustic
        solo_instrument = solo_instrument_acoustic
        if instrument.program == 25 or instrument.program == 26:
            chords_instrument = chords_instrument_electric
            solo_instrument = solo_instrument_electric
        elif instrument.program > 32 or instrument.program < 25:
            continue

        # Sort notes by start time
        sorted_notes = sorted(instrument.notes, key=lambda note: note.start)

        prev_time = None
        simultaneous_notes = []

        for note in sorted_notes:
            if prev_time is None or abs(note.start - prev_time) < 0.05:
                # Notes played very close together are likely part of a chord
                simultaneous_notes.append(note)
            else:
                # Process previous notes
                if len(simultaneous_notes) > 1:
                    chords_instrument.notes.extend(simultaneous_notes)
                else:
                    solo_instrument.notes.extend(simultaneous_notes)

                # Start a new sequence
                simultaneous_notes = [note]

            prev_time = note.start

        # Handle the last set of notes
        if len(simultaneous_notes) > 1:
            chords_instrument.notes.extend(simultaneous_notes)
        else:
            solo_instrument.notes.extend(simultaneous_notes)

    chords_midi.instruments.append(chords_instrument_acoustic)
    chords_midi.instruments.append(chords_instrument_electric)
    solo_midi.instruments.append(solo_instrument_acoustic)
    solo_midi.instruments.append(solo_instrument_electric)


separate_guitar_tracks("input_guitar.mid", "guitar_chords.mid", "guitar_solo.mid")
