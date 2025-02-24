import pretty_midi

def separate_guitar_tracks(midi_path, output_chords, output_solo):
    # Load MIDI file
    midi_data = pretty_midi.PrettyMIDI(midi_path)

    # Create new MIDI files for chords and solo
    chords_midi = pretty_midi.PrettyMIDI()
    solo_midi = pretty_midi.PrettyMIDI()

    for instrument in midi_data.instruments:
        # Skip non-guitar instruments
        instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
        if "Guitar" not in instrument_name and "guitar" not in instrument_name:
            continue

        # New instruments for chords and solo
        chords_instrument = pretty_midi.Instrument(program=instrument.program)
        solo_instrument = pretty_midi.Instrument(program=instrument.program)

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

        # Add instruments to respective MIDI files
        if chords_instrument.notes:
            chords_midi.instruments.append(chords_instrument)
        if solo_instrument.notes:
            solo_midi.instruments.append(solo_instrument)

    # Save new MIDI files
    if chords_midi.instruments:
        chords_midi.write(output_chords)
        print(f"Chords saved to {output_chords}")
    if solo_midi.instruments:
        solo_midi.write(output_solo)
        print(f"Solo saved to {output_solo}")

# Example usage

separate_guitar_tracks("input_guitar.mid", "guitar_chords.mid", "guitar_solo.mid")
