import os

# Features
WIDTH = 64
HEIGHT = 64
NUM_OF_FEATURES = WIDTH * HEIGHT
NUM_OF_CLASSES = 1
MINIBATCH_SIZE = 10000

#INPUT_DATA
IN_PATH = os.path.abspath(os.path.join('..','DATA/RMFD'))
IN_MASKED_FACE_PATH = os.path.join(IN_PATH, 'AFDB_masked_face_dataset')
IN_MASKED_FACE_AUGMENTED_PATH = os.path.join(IN_PATH, 'AFDB_masked_face_dataset_augmented')
IN_FACE_PATH = os.path.join(IN_PATH, 'AFDB_face_dataset')
#test cases
IN_TESTCASE1 = os.path.abspath(os.path.join('..','DATA/TESTCASE1'))

#OUTPUT DATA
OUT_PATH = './DATASET/'
out_testcase1 = OUT_PATH + 'testcase1.npy'
out_masked_dataset = OUT_PATH + 'masked_dataset.npy'
out_face_dataset = OUT_PATH + 'face_dataset.npy'
out_mixed_dataset = OUT_PATH + 'mixed_dataset.npy'
out_mixed_balanced_dataset = OUT_PATH + 'mixed_balanced_dataset.npy' # 53639 samples with balanced distribution.
