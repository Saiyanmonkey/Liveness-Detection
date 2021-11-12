#import
import pydot
import pydotplus
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model


class LivenessNet:
    @staticmethod
    def build(width,height,depth,classes):
        #initialise the models along with the input shape 
        #to be "channels last" and the channel dimension itself
        model = Sequential()
        inputShape = (height,width,depth)
        channelDim = -1
        
        #if use "channel first", update input shape
        #and channel dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth,height,width)
            channelDim = 1
            
        #Adding Layers
        # First CONV ==> RELU ==> CONV ==> RELU ==> POOL
        model.add(Conv2D(16,(3,3),padding="same",input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelDim))
        
        model.add(Conv2D(16,(3,3),padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelDim))
        
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        
        # Second CONV ==> RELU ==> CONV ==> RELU ==> POOL
        #model.add(Conv2D(32,(3,3),padding="same"))
        #model.add(Activation("relu"))
        #model.add(BatchNormalization(axis=channelDim))
        
        #model.add(Conv2D(32,(3,3),padding="same"))
        #model.add(Activation("relu"))
        #model.add(BatchNormalization(axis=channelDim))
        
        #model.add(MaxPooling2D(pool_size=(2,2)))
        #model.add(Dropout(0.25))
        
        # Final CONV ==> RELU ==> CONV ==> RELU ==> POOL
        model.add(Conv2D(32,(3,3),padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelDim))
        
        model.add(Conv2D(32,(3,3),padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelDim))
        
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        
        #first and only set of FC ==> RELU
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        #softmac classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        
        plot_model(model, to_file='C:/Users/alvin/OneDrive/Desktop/Alvin/Kerja/Week 3/liveness-detection-lcc/model_plot.png', show_shapes=True, show_layer_names=True)
        
        return model
        
        
        