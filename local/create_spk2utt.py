import os
from glob import glob


root_dir = "../db/cu-multilang-dataset/"

for fol in glob(root_dir + "*/"):
    if os.path.exists(fol + "spk2utt"):
        os.remove(fol + "spk2utt")
    
    spk2utt = []
    with open(fol + "wav.scp") as infile:
        for line in infile.readlines():
            utt_id = line.split()[0]
            speaker_name = utt_id
            spk2utt.append((speaker_name, utt_id))
            
    with open(fol + "spk2utt", "w") as outfile:
        for spk, utt in spk2utt:
            outfile.write(spk + "    " + utt + "\n")
            
print("Done creating spk2utt.")
