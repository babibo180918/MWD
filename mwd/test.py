import os
import random
import config as cf
import dataset as ds
import Utils


# masked dataset
#ds.create_dataset(cf.IN_MASKED_FACE_PATH, cf.out_masked_dataset, cf.NUM_OF_CLASSES, 1, 1000)

#face dataset
#ds.create_dataset(cf.IN_FACE_PATH, cf.out_face_dataset, cf.NUM_OF_CLASSES, 0, 1000)

'''
#mixed dataset
masked_filepaths = Utils.getFileList(cf.IN_MASKED_FACE_PATH)
masked_labels = [1]*len(masked_filepaths)
face_filepaths = Utils.getFileList(cf.IN_FACE_PATH)
face_labels = [0]*len(face_filepaths)
ds.create_mixed_dataset(cf.out_mixed_dataset, cf.NUM_OF_CLASSES, masked_filepaths, masked_labels, face_filepaths, face_labels)
'''

# mixed balanced dataset
masked_filepaths = Utils.getFileList(cf.IN_MASKED_FACE_AUGMENTED_PATH)
masked_labels = [1]*len(masked_filepaths)
# suffle random masked images
sample_idxs = random.sample(range(0,len(masked_filepaths)), len(masked_filepaths))
masked_filepaths = [masked_filepaths[i] for i in sample_idxs]
masked_labels = [masked_labels[i] for i in sample_idxs]

face_filepaths = Utils.getFileList(cf.IN_FACE_PATH)
face_labels = [0]*len(face_filepaths)
# get 30000 samples of faces
sample_idxs = random.sample(range(0,len(face_filepaths)), 30000)       
face_filepaths = [face_filepaths[i] for i in sample_idxs]
face_labels = [face_labels[i] for i in sample_idxs]
# Create training dataset
ds.create_mixed_dataset(cf.out_mixed_balanced_dataset_train, cf.NUM_OF_CLASSES, masked_filepaths[0:22000], masked_labels[0:22000], face_filepaths[0:28000], face_labels[0:28000])
# Create test dataset
ds.create_mixed_dataset(cf.out_mixed_balanced_dataset_test, cf.NUM_OF_CLASSES, masked_filepaths[22000:len(masked_filepaths)], masked_labels[22000:len(masked_labels)], face_filepaths[28000:len(face_filepaths)], face_labels[28000:len(face_labels)])
