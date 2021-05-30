from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.metrics import *

class MWD_Model(Model):
    '''
    define Model specified for MWD using CNN as the main component
    '''
    def __init__(self, inputs, classes, kernel_initializer=initializers.GlorotNormal(),
                 name='MWD_Model', **kwargs):
        super(MWD_Model, self).__init__()
        self.ZeroPadding0 = ZeroPadding2D((2,2))
        self.Conv0 = Conv2D(64, (5,5), strides=(1,1), name="conv0", activation='relu')
        self.MP2D0 = MaxPooling2D((2,2), name="max_pool0")
        self.Conv1 = Conv2D(128, (3,3), strides=(1,1), name='conv1', activation='relu')
        self.AP2D1 = AveragePooling2D((2,2), name='average_pool1')
        self.Flatten0 = Flatten()
        self.FC0 = Dense(16, activation='relu', name='FC0')
        self.FC1 = Dense(classes, activation='sigmoid', name='FC1')

    
    def call(self, inputs):
        x = self.ZeroPadding0(inputs)
        x = self.Conv0(x)
        x = self.MP2D0(x)
        x = self.Conv1(x)
        x = self.AP2D1(x)
        x = self.Flatten0(x)
        x = self.FC0(x)
        x = self.FC1(x)
        return x
    
    def train_step(self, data):
        return super().train_step(data)
        