import pretty_midi

def notes_overlap(note_a, note_b, min_overlap=0.03):
    """
    Returns True if note_a and note_b overlap in time by at least `min_overlap` seconds.
    """
    overlap_start = max(note_a.start, note_b.start)
    overlap_end = min(note_a.end, note_b.end)
    return (overlap_end - overlap_start) >= min_overlap

def separate_guitar_tracks_improved(
    midi_path,
    output_chords,
    output_solo,
    start_time_threshold=0.05,
    overlap_threshold=0.03
):
    """
    Improved chord detection:
      - Groups notes that start within `start_time_threshold` of each other.
      - Confirms they overlap in time by `overlap_threshold`.
      - If a group has >= 2 notes truly overlapping, it's treated as a chord.
    """
    midi_data = pretty_midi.PrettyMIDI(midi_path)

    chords_midi = pretty_midi.PrettyMIDI()
    solo_midi = pretty_midi.PrettyMIDI()

    chords_instrument = pretty_midi.Instrument(program=28)  # e.g. Electric Guitar (Jazz)
    solo_instrument = pretty_midi.Instrument(program=26)    # e.g. Electric Guitar (Clean)

    for instrument in midi_data.instruments:
        # Filter for guitar-like programs
        if not (25 <= instrument.program <= 32):
            continue

        # Sort notes by start time
        sorted_notes = sorted(instrument.notes, key=lambda n: n.start)

        cluster = []
        prev_start = None

        for note in sorted_notes:
            if prev_start is None:
                cluster = [note]
                prev_start = note.start
                continue

            # If the start of this note is close to the previous note's start,
            # consider it as part of the same cluster
            if abs(note.start - prev_start) <= start_time_threshold:
                cluster.append(note)
            else:
                # Evaluate the cluster we just ended
                if len(cluster) > 1:
                    # Check how many notes truly overlap
                    # If at least 2 overlap in time, we treat the whole cluster as chord
                    overlapping_pairs = 0
                    for i in range(len(cluster)):
                        for j in range(i + 1, len(cluster)):
                            if notes_overlap(cluster[i], cluster[j], min_overlap=overlap_threshold):
                                overlapping_pairs += 1

                    if overlapping_pairs > 0:
                        chords_instrument.notes.extend(cluster)
                    else:
                        # No real overlaps found => treat as single notes
                        solo_instrument.notes.extend(cluster)
                else:
                    # Only one note in cluster => definitely a solo note
                    solo_instrument.notes.extend(cluster)

                # Start a new cluster
                cluster = [note]

            prev_start = note.start

        # Handle the last cluster in the loop
        if cluster:
            if len(cluster) > 1:
                # Check for overlap
                overlapping_pairs = 0
                for i in range(len(cluster)):
                    for j in range(i + 1, len(cluster)):
                        if notes_overlap(cluster[i], cluster[j], min_overlap=overlap_threshold):
                            overlapping_pairs += 1

                if overlapping_pairs > 0:
                    chords_instrument.notes.extend(cluster)
                else:
                    solo_instrument.notes.extend(cluster)
            else:
                solo_instrument.notes.extend(cluster)

    chords_midi.instruments.append(chords_instrument)
    solo_midi.instruments.append(solo_instrument)

    if chords_instrument.notes:
        chords_midi.write(output_chords)
        print(f"Chords saved to {output_chords}")
    if solo_instrument.notes:
        solo_midi.write(output_solo)
        print(f"Solo saved to {output_solo}")

separate_guitar_tracks_improved(
    "input_guitar.mid",
    "improved_chords.mid",
    "improved_solo.mid",
    start_time_threshold=0.05,
    overlap_threshold=0.03
)
