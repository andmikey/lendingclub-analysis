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
shuf -n "$1" "$2" > "$3"
