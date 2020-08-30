import os
import config as cf
import dataset as ds
import Utils


# masked dataset
#ds.create_dataset(cf.IN_MASKED_FACE_PATH, cf.out_masked_dataset, cf.NUM_OF_CLASSES, 1, 1000)
#ds.load_dataset(cf.out_masked_dataset)

#face dataset
#ds.create_dataset(cf.IN_FACE_PATH, cf.out_face_dataset, cf.NUM_OF_CLASSES, 0, 1000)
#ds.load_dataset(cf.out_face_dataset)

#mixed dataset
masked_filepaths = Utils.getFileList(cf.IN_MASKED_FACE_PATH)
masked_labels = [1]*len(masked_filepaths)
face_filepaths = Utils.getFileList(cf.IN_FACE_PATH)
face_labels = [0]*len(face_filepaths)
ds.create_mixed_dataset(cf.out_mixed_dataset, cf.NUM_OF_CLASSES, masked_filepaths, masked_labels, face_filepaths, face_labels)
ds.load_dataset(cf.out_mixed_dataset)
