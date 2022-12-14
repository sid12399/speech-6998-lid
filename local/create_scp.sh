#!/bin/bash
#This bash script will create a text file containing the filenames of all .wav files in the given directory and their absolute paths

#Loop through directory and write absolute paths and filenames of .wav files to output file

for folder in db/cu-multilang-dataset/*; do
        echo "$folder"
        rm $folder/wav.scp
        for filename in $folder/*.wav; do
                echo "$(basename $filename .wav) $(realpath $filename)" >> $folder/wav.scp
        done
done
