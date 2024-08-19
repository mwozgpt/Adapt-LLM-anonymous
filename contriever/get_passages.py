import json
import pandas as pd
from tqdm import tqdm
import os
import faiss
from faiss import read_index
import torch
import pickle
import numpy as np
import sys

# Read Wikipedia data
df = pd.read_csv('psgs_w100_4.tsv', sep='\t', engine="pyarrow")
wikipedia = list(df['text'])
print("Wikipedia length:", len(wikipedia))

# Load pre-built index
index = read_index("wikipedia.index")  # May take hours to load!
print("Total entries in index:", index.ntotal)

batch_size = 64

# Function to split list into batches
def chunks(lst, batch_size):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

# Check command line argument for dataset type
if len(sys.argv) != 2:
    print("Usage: python combined_script.py [nq | squad]")
    sys.exit(1)

arg = sys.argv[1]
if arg not in ["nq", "squad"]:
    print("Invalid argument. Use 'nq' or 'squad'.")
    sys.exit(1)

# Define splits based on dataset type
if arg == "nq":
    splits = ["test", "train"]
elif arg == "squad":
    splits = ["dev", "train"]

# File pattern for loading pickled dataset
pattern = "biencoder-" + arg + "-{}_predictions._pickled.obj"

for split in splits:
    extract_questions_path = os.path.join("new_dataset", pattern.format(split))

    with open(extract_questions_path, "rb") as f:
        dataset = pickle.load(f)
    print(extract_questions_path)
    print(len(dataset))
    list_dataset = [[key, content] for key, content in dataset.items()]

    question_document_pairs = {}
    for c in tqdm(chunks(list_dataset, batch_size), total=len(dataset)//batch_size):
        _vector = np.vstack([content['question_embedding'] for _, content in c])

        # Normalize vectors
        faiss.normalize_L2(_vector)

        # Perform search
        (distances, ann) = index.search(_vector, k=1)

        # Populate question-document pairs
        for k in range(_vector.shape[0]):
            question_document_pairs[c[k][0]] = {
                "question": c[k][1]["question"],
                "possible_answers": c[k][1]["possible_answers"],
                "gold_passage": c[k][1]["gold_passage"],
                "contriever_passage": wikipedia[ann[k][0]],
                "contriver_score": float(distances[k][0]),
            }

    # Save results as JSON
    with open(f"new_dataset/{os.path.basename(extract_questions_path)[:-4]}_predictions.json", "w") as json_file:
        json.dump(question_document_pairs, json_file)
