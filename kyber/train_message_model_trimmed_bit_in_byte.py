import tensorflow as tf
import os.path
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.layers import Dense, InputLayer, BatchNormalization, ReLU, Softmax, Flatten, Conv1D, AveragePooling1D, Dropout, LSTM
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.optimizers import Nadam, RMSprop, SGD
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras.models import load_model
from keras.callbacks import EarlyStopping
import time
from scipy import stats
import fnmatch
from tqdm import trange

################################################
data_folder = "traces/profiling/"

bit_in_byte = 7 # 0 or 7

trimmask_name = f"message_bit{bit_in_byte}_trim_mask.npy"

model_folder = f"models/message/25k_trim1/bit_in_byte_{bit_in_byte}/"
modelName = 'kyber_message_'

NUM_OF_TRACES = 25000
#################################################

def create_model(classes=256, input_size=800):
    input_shape = (input_size,)

    # Create model.
    model = Sequential()    
    model.add(BatchNormalization(input_shape=input_shape))

    model.add(Dense(128, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(Dense(64, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(Dense(32, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(Dense(classes, kernel_initializer='he_uniform'))
    model.add(Softmax())

    learning_rate = 0.001
    print('learning_rate =', learning_rate)

    optimizer = Nadam(lr=learning_rate, epsilon=1e-08)  

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    print(model.summary())
    return model

def hamming_weight(n):
    return bin(np.uint8(n)).count("1")

def t_test(trace_0,trace_1):

    print(len(trace_0),len(trace_1))

    (statistic,pvalue) = stats.ttest_ind(trace_0, trace_1, equal_var = False)

    plt.plot(statistic)
    plt.ylabel('SOST value')
    plt.xlabel('trace point')

    m_index = statistic.argsort()[-5:][::-1]
    tmp = statistic[m_index][0]
    print(m_index, 'max positive SOST =', tmp)

    m_index = statistic.argsort()[0:5][::]
    tmp = statistic[m_index][0]
    print(m_index, 'max negative SOST =', tmp)

    return

def check_file_exists(file_path):
    if os.path.exists(file_path) == False:
        print("Error: provided file path '%s' does not exist!" % file_path)
        sys.exit(-1)
    return

def return_kth_bit(n,k):
    return((n & (1 << k)) >> k)

#def load_traces():
#    #Import training traces and labels
#
#    trimmask = np.load(trimmask_name)
#
#    untrimmed_traces = np.load(data_folder+'cut_joined_traces.npy', mmap_mode="r")[:NUM_OF_TRACES*253]
#    labels = np.load(data_folder+'cut_joined_message_labels.npy', mmap_mode="r")[:NUM_OF_TRACES*253]
#    shuffle_labels = np.load(data_folder+'cut_joined_shuffle_labels.npy', mmap_mode="r")[:NUM_OF_TRACES*253]
#    #traces = np.load(data_folder+'cut_traces_0.npy', mmap_mode="r")[:,450+210:-210]
#    #labels = np.load(data_folder+'cut_message_labels_0.npy', mmap_mode="r")[:]
#    #print(labels[:100])
#    #[plt.plot(trace) for trace in traces[:100]]
#    #plt.show()
#
#    traces = np.empty((NUM_OF_TRACES*253, len(trimmask)))
#    print("Trimming using mask:", trimmask)
#    for i in trange(NUM_OF_TRACES*253):
#        traces[i] = untrimmed_traces[i, trimmask]
#
#    print("Shape of all traces",traces.shape)    
#    print("Shape of all labels",labels.shape)
#
#    traces = traces[np.where(shuffle_labels % 8 == bit_in_byte)]
#    labels = labels[np.where(shuffle_labels % 8 == bit_in_byte)]
#
#    print("Shape of bit-in-byte traces",traces.shape)    
#    print("Shape of bit-in-byte labels",labels.shape)
#
#    np.save(f"traces/bit_shuffled/bit{bit_in_byte}_traces.npy", traces)
#    np.save(f"traces/bit_shuffled/bit{bit_in_byte}_labels.npy", labels)
#    sys.exit()
#
#    return traces, labels

def load_traces():
    traces = np.load(f"traces/profiling/bit{bit_in_byte}_traces.npy")
    labels = np.load(f"traces/profiling/bit{bit_in_byte}_labels.npy")

    return traces, labels

def train_model(X_profiling, Y_profiling, model, save_file_name, i, epochs=100, batch_size=32, patience=50, classes=2):  
    check_file_exists(os.path.dirname(save_file_name))
    # Save model every epoch
    save_model = ModelCheckpoint(save_file_name+str(i)+'.h5',monitor='val_accuracy',verbose=1,save_best_only=True,mode='max')
    # Get the input layer shape
    input_layer_shape = model.get_layer(index=0).input_shape
    # Sanity check
    if input_layer_shape[1] != len(X_profiling[0]):
        print("Error: model input shape %d instead of %d is not expected ..." % (input_layer_shape[1], len(X_profiling[0])))
        sys.exit(-1)
    # Adapt the data shape according our model input
    if len(input_layer_shape) == 2:
        # This is a MLP
        Reshaped_X_profiling = X_profiling
    elif len(input_layer_shape) == 3:
        # This is a CNN: expand the dimensions
        Reshaped_X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1))
    else:
        print("Error: model input shape length %d is not expected ..." % len(input_layer_shape))
        sys.exit(-1)

    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=patience)

    history = model.fit(x=Reshaped_X_profiling, y=to_categorical(Y_profiling, num_classes=classes), batch_size=batch_size, verbose=1, epochs=epochs, callbacks=[es,save_model], validation_split=0.3)

    # summarize history for accuracy
    plt.clf()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f"history/message/bit_in_byte_{bit_in_byte}/" +modelName+str(i)+'acc.pdf')

    # summarize history for loss
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f"history/message/bit_in_byte_{bit_in_byte}/" +modelName+str(i)+'loss.pdf')

    return history
    
# Start of execution, the time parts are there for our own references so we know roughly how long training takes
start = time.time()

# Load the profiling traces 
(traces, labels) = load_traces()

input_size = traces.shape[1]

print('input size =', input_size)

model_number = -1
for model_name in os.listdir(model_folder):
    if fnmatch.fnmatch(model_name, "kyber_message_[0-9]*.h5"):
        n = int(model_name[14:-3])
        if n > model_number: model_number = n
model_number += 1

for i in range(model_number,10):  
    print('i =', i)
    print('bit in byte =', bit_in_byte)

    ### MLP training from scratch
    mlp = create_model(classes=2, input_size=input_size)

    train_model(traces, labels, mlp, model_folder + modelName, i, epochs=500, batch_size=128, patience=30, classes=2)

end = time.time()

print("The total running time was: ",((end-start)/60), " minutes.") 
