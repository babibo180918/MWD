import os
import config as cf
import dataset as ds


# masked dataset
#ds.create_dataset(cf.IN_MASKED_FACE_PATH, cf.out_masked_dataset, cf.NUM_OF_CLASSES, 1, 1000)
#X, y, filepaths = ds.load_dataset(cf.out_masked_dataset)

#face dataset
ds.create_dataset(cf.IN_FACE_PATH, cf.out_face_dataset, cf.NUM_OF_CLASSES, 0, 1000)
X, y, filepaths = ds.load_dataset(cf.out_face_dataset)

print('X.shape = ' + str(X.shape))
print('y.shape = ' + str(y.shape))
print('filepaths.shape = ' + str(filepaths.shape))
