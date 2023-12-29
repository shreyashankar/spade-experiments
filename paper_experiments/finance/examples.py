import os
import pandas as pd

char_cutoff = 15000

# Get current file's parent directory
parent_dir = os.path.dirname(os.path.abspath(__file__))

# Get dataset directory
dataset_dir = os.path.join(parent_dir, "Transcripts")

# Recursively get all text files in the dataset directory
transcript_files = []
for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith(".txt"):
            transcript_files.append(os.path.join(root, file))

# Read all transcripts
transcripts = []
for file in transcript_files:
    with open(file, "r") as f:
        transcript = f.read()
        transcripts.append(transcript[:15000])

# Get 50 transcripts
transcripts = transcripts[:50]

EXAMPLES = [{"docs": doc} for doc in transcripts]
