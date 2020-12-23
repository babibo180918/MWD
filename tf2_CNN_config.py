import os

# Features
WIDTH = 64
HEIGHT = 64
NUM_OF_FEATURES = WIDTH * HEIGHT
NUM_OF_CLASSES = 1
MAX_BATCH_SIZE = 2048

#INPUT_DATA
IN_PATH = os.path.abspath(os.path.join('../../..','DATA/RMFD'))
IN_MASKED_FACE_PATH = os.path.join(IN_PATH, 'AFDB_masked_face_dataset')
IN_MASKED_FACE_AUGMENTED_PATH = os.path.join(IN_PATH, 'AFDB_masked_face_dataset_augmented')
IN_FACE_PATH = os.path.join(IN_PATH, 'AFDB_face_dataset')

#OUTPUT DATA
OUT_PATH = './DATASET/'
out_mixed_balanced_dataset = OUT_PATH + '3c_mixed_balanced_dataset.npy' # 53639 samples with balanced distribution.
out_mixed_balanced_dataset_train = OUT_PATH + '3c_mixed_balanced_dataset_train.npy' # 32768 samples with balanced distribution.
out_mixed_balanced_dataset_validate = OUT_PATH + '3c_mixed_balanced_dataset_val.npy' # 10240 samples with balanced distribution.
out_mixed_balanced_dataset_test = OUT_PATH + '3c_mixed_balanced_dataset_test.npy' # 10631 samples with balanced distribution.
