import time
import fluidsynth

fs = fluidsynth.Synth()
fs.start()

sfid = fs.sfload("electric.sf2")
fs.program_select(0, sfid, 0, 0)
fs.setting("synth.gain", 1.5)  

for i in range(45, 100):
    print(i)
    fs.noteon(0, i, 64)

    time.sleep(1)
    fs.noteoff(0, i)

    time.sleep(0)

fs.delete()