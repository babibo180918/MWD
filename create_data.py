import os
import random
import gdown
import zipfile
from zipfile import ZipFile
import shutil
import pathlib
from tqdm import tqdm

import numpy as np

from mwd import Utils

DATASET_URL = "https://drive.google.com/uc?export=download&id=1iVCRw2YOzkYRxDhafL7-BZjIKN6uEbHM"
DATA_PATH = os.path.abspath("./data")
DATA_FILE = os.path.join(DATA_PATH, 'RMFD.zip')
MASKED_FACE_PATH = os.path.join(DATA_PATH, 'RMFD/AFDB_masked_face_dataset')
FACE_PATH = os.path.join(DATA_PATH, 'RMFD/AFDB_face_dataset')
MASKED_FACE_AUGMENTED_PATH = os.path.join(DATA_PATH, 'RMFD/AFDB_masked_face_dataset_augmented')
FACE_SAMPLE_NUM = 30000
MASKED_SAMPLE_NUM = 30000
DATASET_PATH = os.path.join(DATA_PATH, 'mixed_balanced_dataset.npy') # 60000 samples with balanced distribution.)

def create_dataset(out_path, *args):
    filepaths = []
    labels = []    
    numOfArgs = len(args)
    
    if numOfArgs%2 != 0:
        print("Number of arguments are incorrect.")
        return

    for i in range(0, numOfArgs, 2):
        filepaths.extend(args[i])
        labels.extend(args[i+1])

    numOfDataset = numOfArgs//2
    sumOfSamples = len(labels)
    sample_idxs = random.sample(range(0,sumOfSamples), sumOfSamples)         
    filepaths = [filepaths[i] for i in sample_idxs]
    labels = [labels[i] for i in sample_idxs]
    inputs = {'filepaths':filepaths, 'labels':labels}
    np.save(out_path, inputs)
    return (filepaths, labels)

#pathlib.Path(DATA_PATH).mkdir(parents=True, exist_ok=True)
#gdown.download(DATASET_URL, DATA_FILE, quiet=False)
'''    
with ZipFile(DATA_FILE,"r") as zip_ref:
    for file in tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist())):
         zip_ref.extract(member=file, path=DATA_PATH)
'''
    
#shutil.copytree(MASKED_FACE_PATH, MASKED_FACE_AUGMENTED_PATH, dirs_exist_ok=True)

# create mixed balanced dataset
face_filepaths = Utils.getFileList(FACE_PATH)
face_labels = [0]*len(face_filepaths)
sample_idxs = random.sample(range(0,len(face_filepaths)), min(len(face_filepaths),FACE_SAMPLE_NUM))      
face_filepaths = [face_filepaths[i] for i in sample_idxs]
face_labels = [face_labels[i] for i in sample_idxs]

masked_filepaths = Utils.getFileList(MASKED_FACE_AUGMENTED_PATH)
masked_labels = [1]*len(masked_filepaths)
sample_idxs = random.sample(range(0,len(masked_filepaths)), min(len(masked_filepaths),MASKED_SAMPLE_NUM))
masked_filepaths = [masked_filepaths[i] for i in sample_idxs]
masked_labels = [masked_labels[i] for i in sample_idxs]

filepaths, labels = create_dataset(DATASET_PATH, masked_filepaths, masked_labels, face_filepaths, face_labels)

print(filepaths[1:5])
print(len(labels))