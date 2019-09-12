#!/bin/bash
# Sample n_samples from the dataset in source_file and save to dest_file.
# Useful if you don't have enough memory to process the entire dataset. 

programname=$0
n_samples=$1
src_file=$2
dst_file=$3

function usage {
    echo "usage: $programname n_samples source_file dest_file"
    exit 1
}

# Must have 3 arguments passed
if [ "$#" -ne 3 ]; then
    usage
fi
# Write header to file
head -n 1 "$2" > "$3"
# Skip shuffling header and sample n_samples
tail -n +2 "$2" | shuf -n "$1" >> "$3"
