import time
import fluidsynth

fs = fluidsynth.Synth()
fs.start()

sfid = fs.sfload("acoustic.sf2")
fs.program_select(0, sfid, 0, 0)
fs.setting("synth.gain", 1.5)  

for i in range(40, 85):
    print(i)
    fs.noteon(0, i, 64)

    time.sleep(1)
    fs.noteoff(0, i)

    time.sleep(0)

fs.delete()