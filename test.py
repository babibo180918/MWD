import os
import config as cf
import dataset as ds

# test
ds.create_dataset(cf.IN_FACE_PATH, cf.out_face_dataset, cf.NUM_OF_CLASSES, 0, 1000)
X, y, filepaths = ds.load_dataset(cf.out_face_dataset)
print('X.shape = ' + str(X.shape))
print('y.shape = ' + str(y.shape))
print('filepaths.shape = ' + str(filepaths.shape))
