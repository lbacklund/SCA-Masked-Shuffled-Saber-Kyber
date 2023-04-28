import os.path
import sys
import h5py
import random
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import time
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D, AveragePooling1D
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.optimizers import Nadam, RMSprop
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras.models import load_model
import os
import fnmatch
from tqdm import tqdm

sys.path.append("..")
from ECC_CCT_tool.ECC_CCT_tool import ECC_CCT_TOOL

##################################################
index_model_folder = "models/index/15k/"        
message_model_folder = "models/message/25k/"
message_bit0_model_folder = "models/message/50k/bit_in_byte_0/"        
message_bit7_model_folder = "models/message/50k/bit_in_byte_7/"        
beginning_index_model_folder = "models/index_beginning/25k/"        
beginning_message_model_folder = "models/message_beginning/25k/"        
end_index_model_folder = "models/index_end/25k/"
end_message_model_folder = "models/message_end/25k/"        

data_folder_prefix = "traces/attack/"
truth_folder = data_folder_prefix+"GNDTruth/"

SAVE_STATISTICS_FILE = "attack_statistics.txt"

#SETS = None
SETS = 1 #10

CODE_DISTANCE = 8
REPS = 30

MAX_NUM_OF_MODELS = 10
VOTE_THRESHOLD = 0.9
BIT_IN_BYTE_MODELS = True
PREDICT_INDEXES = [0, 255]

################################################### 

# Initialize ECC-CCT tool
ECC_tool = ECC_CCT_TOOL("kyber768", CODE_DISTANCE)

data_folder = data_folder_prefix+f"cd{CODE_DISTANCE}/"

def synchronize_shuffle_traces(traces):
    print("Synchronizing shuffle traces")
    
    #for trace in traces[:10]:
    #    plt.plot(trace)
    #plt.show()

    pattern = np.mean(traces[:, 290:345-15], axis=0) # Mean of all second spikes

    correlations = [np.correlate(trace, pattern) for trace in tqdm(traces, "Correlating traces")]
    
    margin = 45
    seg_len = 55+2*margin
    new_traces = np.empty((traces.shape[0], seg_len*255))
    c = 0
    for i in tqdm(range(traces.shape[0]), "Synchronizing traces"): # process every trace
        trace = traces[i]
        correlation = correlations[i]
        peak_counter = 255
        threshold = 0.05
        n = 230
        while n < traces.shape[1]-len(pattern): # Go through all points in shuffling
            try:
                if correlation[n] > threshold:
                    peak_index = n + np.argmax(correlation[n:n+10])
                    new_traces[c, seg_len*(255-peak_counter):seg_len*(256-peak_counter)] = trace[peak_index-margin:peak_index+seg_len-margin]
                    n = peak_index + 48
                    peak_counter -= 1
                    if not peak_counter: break
                else:
                    n += 1
            except Exception as e:
                plt.plot(correlations[i])
                plt.axhline(y=threshold, color='red')
                plt.show()
                raise e
        if peak_counter:
            print(str(peak_counter)+" shuffling peaks not found!")
            plt.plot(correlations[i])
            plt.axhline(y=threshold, color='red')
            plt.show()
        c += 1
    
    #for trace in new_traces[:20]:
    #    plt.plot(trace)
    #plt.show()
   
    return new_traces

def synchronize_message_traces(traces):
    print("Synchronizing message traces")

    #for trace in traces[:10]:
    #    plt.plot(trace)
    #plt.show()

    end_pattern = np.mean(traces[:, 58150:58400], axis=0)
    end_correlations = [np.correlate(trace, end_pattern) for trace in tqdm(traces, "Correlating end of traces")]
    bad_traces = []
    for i in range(traces.shape[0]):
        if np.max(end_correlations[i][1000:58100]) > 0.5:
            bad_traces.append(i)
            print(i, "bad")
    if len(bad_traces):
        print(len(bad_traces), "out of", traces.shape[0], "incorrectly cut")

    pattern = np.mean(traces[:, 900:1125-100], axis=0) # Mean of all second spikes

    correlations = [np.correlate(trace, pattern) for trace in tqdm(traces, "Correlating traces")]
    
    seg_len = 225
    new_traces = np.empty((traces.shape[0], seg_len*256))
    c = 0
    for i in tqdm(range(traces.shape[0]), "Synchronizing traces"): # process every trace
        trace = traces[i]
        correlation = correlations[i]
        shift = -60
        threshold = 0.045
        n = 640
        peak_counter = 256
        while True:
            if correlation[n] > threshold:
                peak_index = n + np.argmax(correlation[n:n+40])
                new_traces[c, seg_len*(256-peak_counter):seg_len*(257-peak_counter)] = trace[peak_index+shift:peak_index+seg_len+shift]
                n = peak_index + 190
                peak_counter -= 1
                if not peak_counter: break
            else: n += 1
            if n >= len(correlation): break
        if peak_counter:
            print(str(peak_counter)+" message peaks not found!")
            plt.plot(correlations[i])
            plt.axhline(y=threshold, color='red')
            plt.show()
        c += 1

    #for trace in new_traces[:20]:
    #    plt.plot(trace)
    #plt.show()
    
    return new_traces

def standardize_traces(traces):
    means = np.mean(np.reshape(traces, (np.prod(traces.shape[:-1]), traces.shape[-1])), axis=0)
    stddevs = np.std(np.reshape(traces, (np.prod(traces.shape[:-1]), traces.shape[-1])), axis=0)
    traces = np.reshape((np.reshape(traces, (np.prod(traces.shape[:-1]), traces.shape[-1]))-means)/stddevs, traces.shape)

    return traces

#def trim_traces(index_traces, message_traces):
#    index_trim_mask = np.load("index_trim_mask.npy")
#    message_trim_mask = np.load("message_trim_mask.npy")-660
#
#    index_traces = np.reshape(np.reshape(index_traces, (index_traces.shape[0]*index_traces.shape[1], index_traces.shape[2]))[:, index_trim_mask], (index_traces.shape[0], index_traces.shape[1], index_trim_mask.shape[0]))
#    message_traces = np.reshape(np.reshape(message_traces, (message_traces.shape[0]*message_traces.shape[1], message_traces.shape[2]))[:, message_trim_mask], (message_traces.shape[0], message_traces.shape[1], message_trim_mask.shape[0]))
#
#    return index_traces, message_traces

def load_models(model_folder, model_file, max_num=None):
    models = []
    for file_name in os.listdir(model_folder):
        if fnmatch.fnmatch(file_name, model_file):
            model = load_model(model_folder+file_name)
            print("Loading:", file_name)
            #print(model.summary())
            models.append(model)
            if max_num and len(models) >= max_num: break
        else:
            continue
    print("Loaded a total of", len(models), "from", model_folder)
    return models

def get_predictions(set, models, traces, part, CT_num, k1, k0, used_traces):

    # Unpack all models
    if models != None:
        index_models = models["index_models"]
        if BIT_IN_BYTE_MODELS:
            message_bit0_models = models["message_bit0_models"]
            message_bit7_models = models["message_bit7_models"]
        else:
            message_models = models["message_models"]
        last_index_models = models["last_index_models"]
        last_message_models = models["last_message_models"]
        first_index_models = models["first_index_models"]
        first_message_models = models["first_message_models"]
    else:
        index_models = None
        if BIT_IN_BYTE_MODELS:
            message_bit0_models = None
            message_bit7_models = None
        else:
            message_models = None
        last_index_models = None
        last_message_models = None
        first_index_models = None
        first_message_models = None

    # Unpack all traces
    if traces != None:
        index_traces = traces["index_traces"][CT_num]
        message_traces = traces["message_traces"][CT_num]
        last_index_traces = traces["last_index_traces"][CT_num]
        last_message_traces = traces["last_message_traces"][CT_num]
        first_index_traces = traces["first_index_traces"][CT_num]
        first_message_traces = traces["first_message_traces"][CT_num]
    else:
        index_traces = None
        message_traces = None
        last_index_traces = None
        last_message_traces = None
        first_index_traces = None
        first_message_traces = None
    
    stacks = np.zeros((128*used_traces, 256))

    prediction_folder = f"predictions/set_{set}/Part{part}_CT{k1}-{k0}/"
    try:
        os.mkdir(prediction_folder[:-1])
    except FileExistsError:
        pass

    keep_indexes = []
    for i in range(128):
        for j in range(used_traces):
            keep_indexes.append(i*REPS+j)

    if traces != None:
        print("last trace shapes:", last_index_traces.shape, last_message_traces.shape)
    
    # Predict the last index of the permutation
    if os.path.exists(prediction_folder+"last_index.npy"):
        last_index_predictions = np.load(prediction_folder+"last_index.npy")
    else:
        last_index_predictions = np.empty((len(last_index_models), last_index_traces.shape[0], 256))
        for m in tqdm(range(len(last_index_models)), "Predicting last indexes"):
            model = last_index_models[m]
            last_index_predictions[m] = model.predict(last_index_traces)
        last_index_predictions = np.sum(np.log(last_index_predictions), axis=0)
        last_index_predictions = np.argmax(last_index_predictions, axis=1)
        np.save(prediction_folder+"last_index.npy", last_index_predictions)
    last_index_predictions = last_index_predictions[keep_indexes]

    # Predict the last message bit processed
    if os.path.exists(prediction_folder+"last_message.npy"):
        last_message_predictions = np.load(prediction_folder+"last_message.npy")
    else:
        last_message_predictions = np.empty((len(last_message_models), last_message_traces.shape[0], 2))
        for m in tqdm(range(len(last_message_models)), "Predicting last message bits"):
            model = last_message_models[m]
            last_message_predictions[m] = model.predict(last_message_traces) 
        last_message_predictions = np.sum(np.log(last_message_predictions), axis=0)
        last_message_predictions = np.argmax(last_message_predictions, axis=1)
        last_message_predictions = np.where(last_message_predictions==0, -1, last_message_predictions)
        np.save(prediction_folder+"last_message.npy", last_message_predictions)
    last_message_predictions = last_message_predictions[keep_indexes]

    if traces != None:
        print("first trace shapes:", first_index_traces.shape, first_message_traces.shape)
    
    # Predict the first index of the permutation (of the generated ones, i.e. index 1)
    if os.path.exists(prediction_folder+"first_index.npy"):
        first_index_predictions = np.load(prediction_folder+"first_index.npy")
    else:
        first_index_predictions = np.empty((len(first_index_models), first_index_traces.shape[0], 256))
        for m in tqdm(range(len(first_index_models)), "Predicting first indexes"):
            model = first_index_models[m]
            first_index_predictions[m] = model.predict(first_index_traces)
        first_index_predictions = np.sum(np.log(first_index_predictions), axis=0)
        first_index_predictions = np.argmax(first_index_predictions, axis=1)
        np.save(prediction_folder+"first_index.npy", first_index_predictions)
    first_index_predictions = first_index_predictions[keep_indexes]

    # Predict the first message bit processed
    if os.path.exists(prediction_folder+"first_message.npy"):
        first_message_predictions = np.load(prediction_folder+"first_message.npy")
    else:
        first_message_predictions = np.empty((len(first_message_models), first_message_traces.shape[0], 2))
        for m in tqdm(range(len(first_message_models)), "Predicting first message bits"):
            model = first_message_models[m]
            first_message_predictions[m] = model.predict(first_message_traces) 
        first_message_predictions = np.sum(np.log(first_message_predictions), axis=0)
        first_message_predictions = np.argmax(first_message_predictions, axis=1)
        first_message_predictions = np.where(first_message_predictions==0, -1, first_message_predictions)
        np.save(prediction_folder+"first_message.npy", first_message_predictions)
    first_message_predictions = first_message_predictions[keep_indexes]

    if traces != None:
        print("trace shapes:", index_traces.shape, message_traces.shape)
    
    # Predict common index traces
    if os.path.exists(prediction_folder+"index.npy"):
        index_predictions = np.load(prediction_folder+"index.npy")
    else:
        index_predictions = np.empty((len(index_models), index_traces.shape[0], 253, 256))
        for m in tqdm(range(len(index_models)), "Predicting indexes"):
            model = index_models[m]
            index_predictions[m] = np.reshape(model.predict(np.reshape(index_traces,
                (index_traces.shape[0]*index_traces.shape[1], index_traces.shape[2]))),
                (index_traces.shape[0], index_traces.shape[1], 256))
        index_predictions = np.sum(np.log(index_predictions), axis=0)
        index_predictions = np.argmax(index_predictions, axis=2)
        index_predictions = index_predictions[:, ::-1]
        np.save(prediction_folder+"index.npy", index_predictions)
    index_predictions = index_predictions[keep_indexes]

    # Predict common message bit traces
    if BIT_IN_BYTE_MODELS:
        if os.path.exists(prediction_folder+"message_bit0.npy"):
            message_bit0_predictions = np.load(prediction_folder+"message_bit0.npy")
        else:
            message_bit0_predictions = np.empty((len(message_bit0_models), 128*REPS, 254, 2))
            for m in tqdm(range(len(message_bit0_models)), "Predicting message bits (bit in byte: 0)"):
                model_bit0 = message_bit0_models[m]
                #for t in range(index_traces.shape[0]):
                #    message_bit0_predictions[m, t, 0] = model_bit0.predict(message_traces[t, 0:1])
                #    for i, si in enumerate(index_predictions[t]):
                #        if si not in PREDICT_INDEXES: continue
                #        message_bit0_predictions[m, t, i+1] = model_bit0.predict(message_traces[t, i+1:i+2])
                message_bit0_predictions[m] = np.reshape(model_bit0.predict(np.reshape(message_traces,
                    (message_traces.shape[0]*message_traces.shape[1], message_traces.shape[2]))),
                    (message_traces.shape[0], message_traces.shape[1], 2))
            message_bit0_predictions_probs = np.max(message_bit0_predictions, axis=3)
            message_bit0_predictions = np.argmax(message_bit0_predictions, axis=3)
            message_bit0_predictions = np.where(message_bit0_predictions_probs < VOTE_THRESHOLD, 2, message_bit0_predictions)
            message_bit0_predictions = np.apply_along_axis(lambda x: np.bincount(x, minlength=3), 0, message_bit0_predictions)
            message_bit0_predictions = message_bit0_predictions[1] - message_bit0_predictions[0]
            np.save(prediction_folder+"message_bit0.npy", message_bit0_predictions)
        message_bit0_predictions = message_bit0_predictions[keep_indexes]
        if os.path.exists(prediction_folder+"message_bit7.npy"):
            message_bit7_predictions = np.load(prediction_folder+"message_bit7.npy")
        else:
            message_bit7_predictions = np.empty((len(message_bit7_models), 128*REPS, 254, 2))
            for m in tqdm(range(len(message_bit7_models)), "Predicting message bits (bit in byte: 7)"):
                model_bit7 = message_bit7_models[m]
                #for t in range(index_traces.shape[0]):
                #    message_bit7_predictions[m, t, 0] = model_bit7.predict(message_traces[t, 0:1])
                #    for i, si in enumerate(index_predictions[t]):
                #        if si not in PREDICT_INDEXES: continue
                #        message_bit7_predictions[m, t, i+1] = model_bit7.predict(message_traces[t, i+1:i+2])
                message_bit7_predictions[m] = np.reshape(model_bit7.predict(np.reshape(message_traces,
                    (message_traces.shape[0]*message_traces.shape[1], message_traces.shape[2]))),
                    (message_traces.shape[0], message_traces.shape[1], 2))
            message_bit7_predictions_probs = np.max(message_bit7_predictions, axis=3)
            message_bit7_predictions = np.argmax(message_bit7_predictions, axis=3)
            message_bit7_predictions = np.where(message_bit7_predictions_probs < VOTE_THRESHOLD, 2, message_bit7_predictions)
            message_bit7_predictions = np.apply_along_axis(lambda x: np.bincount(x, minlength=3), 0, message_bit7_predictions)
            message_bit7_predictions = message_bit7_predictions[1] - message_bit7_predictions[0]
            np.save(prediction_folder+"message_bit7.npy", message_bit7_predictions)
        message_bit7_predictions = message_bit7_predictions[keep_indexes]
    else:
        if os.path.exists(prediction_folder+"message.npy"):
            message_predictions = np.load(prediction_folder+"message.npy")
        else:
            message_predictions = np.empty((len(message_models), 128*REPS, 254, 2))
            for m in tqdm(range(len(message_models)), "Predicting message bits"):
                model = message_models[m]
                message_predictions[m] = np.reshape(model.predict(np.reshape(message_traces,
                    (message_traces.shape[0]*message_traces.shape[1], message_traces.shape[2]))),
                    (message_traces.shape[0], message_traces.shape[1], 2))
            #############################
            # Basic prediction method
            #message_predictions = np.sum(np.log(message_predictions), axis=0)
            #message_predictions = np.argmax(message_predictions, axis=2)
            #message_predictions = np.where(message_predictions==0, -1, message_predictions)
            #############################
            # Threshold prediction method
            message_predictions_probs = np.max(message_predictions, axis=3)
            message_predictions = np.argmax(message_predictions, axis=3)
            message_predictions = np.where(message_predictions_probs < VOTE_THRESHOLD, 2, message_predictions)
            message_predictions = np.apply_along_axis(lambda x: np.bincount(x, minlength=3), 0, message_predictions)
            message_predictions = message_predictions[1] - message_predictions[0]
            np.save(prediction_folder+"message.npy", message_predictions)
        message_predictions = message_predictions[keep_indexes]

    # Write all predictions made on last index, to stack
    for t in range(128*used_traces):
        stacks[t, last_index_predictions[t]] = last_message_predictions[t]

    # Write all predictions made on first index generated, index 1, to stack
    for t in range(128*used_traces):
        if BIT_IN_BYTE_MODELS:
            if first_index_predictions[t] % 8 == 0:
                stacks[t, first_index_predictions[t]] = message_bit0_predictions[t, 0]
            elif first_index_predictions[t] % 8 == 7:
                stacks[t, first_index_predictions[t]] = message_bit7_predictions[t, 0]
        else:
            stacks[t, first_index_predictions[t]] = message_predictions[t, 0]

    # If 0 or 255 not in predicted indexes, it is likely index 0
    for t in range(128*used_traces):
        if 0 not in index_predictions[t] + [last_index_predictions[t]] + [first_index_predictions[t]]:
            stacks[t, 0] = first_message_predictions[t]
        elif 255 not in index_predictions[t] + [last_index_predictions[t]] + [first_index_predictions[t]]:
            stacks[t, 255] = first_message_predictions[t]

    # Write all common predictions to stack
    for t in range(128*used_traces):
        if BIT_IN_BYTE_MODELS:
            for i, (bit0, bit7) in enumerate(zip(message_bit0_predictions[t, 1:], message_bit7_predictions[t, 1:])):
                if index_predictions[t, i] % 8 == 0:
                    stacks[t, index_predictions[t, i]] = bit0
                elif index_predictions[t, i] % 8 == 7:
                    stacks[t, index_predictions[t, i]] = bit7
        else:
            for i, bit in enumerate(message_predictions[t, 1:]):
                stacks[t, index_predictions[t, i]] = bit

    if traces != None:
        print("trace shapes:", index_traces.shape, message_traces.shape)

    # Ignore all but selected bit indexes
    for i in range(256):
        if i not in PREDICT_INDEXES:
            stacks[:, i] = np.zeros((128*used_traces))

    # Flip positions 0 and 255
    stacks = np.flip(stacks, axis=1)
    
    # Shift stacks to compensate for rotation
    for i, rot in enumerate(range(1,256,2)):
        stacks[i*used_traces:(i+1)*used_traces] = np.roll(stacks[i*used_traces:(i+1)*used_traces], shift=rot, axis=1)

    return stacks

def load_traces(filenum, part):
    print("Loading traces")

    # Derive how many repetitions were captured for each rotation of each message
    captured_reps = int(np.load(data_folder+f"shuffle_traces_part0_CT{ECC_tool.ct_table[0][0]}-{ECC_tool.ct_table[0][1]}_{filenum}.npy", mmap_mode="r").shape[0]/128)
    if REPS > captured_reps:
        print("Not enough repetitions captured. Exiting")
        sys.exit()

    # Find the size of trimmed traces
    index_len = 145+3*225 #len(np.load("index_trim_mask.npy"))
    message_len = 225 #len(np.load("message_trim_mask.npy"))

    # Initialize arrays for all traces
    all_index_traces = np.zeros((len(ECC_tool.ct_table), 128*REPS, 253, index_len), dtype=np.float32)
    all_message_traces = np.zeros((len(ECC_tool.ct_table),128*REPS, 254, message_len), dtype=np.float32)
    first_index_traces = np.zeros((len(ECC_tool.ct_table), 128*REPS, 145+3*225), dtype=np.float32)
    last_index_traces = np.zeros((len(ECC_tool.ct_table), 128*REPS, 145+2*225), dtype=np.float32)
    first_message_traces = np.zeros((len(ECC_tool.ct_table),128*REPS, 225), dtype=np.float32)
    last_message_traces = np.zeros((len(ECC_tool.ct_table),128*REPS, 225), dtype=np.float32)

    # Process traces for each message
    for CT_num, (k1, k0, k2) in enumerate(ECC_tool.ct_table):
        print(f"Loading traces for: part-{part}, k1-{k1}, k0-{k0}, k2-{k2}")

        # Load the traces
        message_trace_filename = f"message_traces_part{part}_CT{k1}-{k0}_{filenum}.npy"
        index_trace_filename = f"shuffle_traces_part{part}_CT{k1}-{k0}_{filenum}.npy"
        raw_message_traces = np.load(data_folder+message_trace_filename)
        raw_index_traces = np.load(data_folder+index_trace_filename)

        # Remove all repeated traces that will not be used
        message_traces = np.zeros((REPS*128, raw_message_traces.shape[1]))
        index_traces = np.zeros((REPS*128, raw_index_traces.shape[1]))
        for i in range(128):
            message_traces[i*REPS:(i+1)*REPS] = raw_message_traces[i*captured_reps:i*captured_reps+REPS]
            index_traces[i*REPS:(i+1)*REPS] = raw_index_traces[i*captured_reps:i*captured_reps+REPS]

        # Synchronize the traces
        synchronized_message_traces = synchronize_message_traces(message_traces)
        synchronized_index_traces = synchronize_shuffle_traces(index_traces)

        # Find the sizes of segments in synchronized traces
        index_seg_size = int(synchronized_index_traces.shape[1]/255)
        message_seg_size = int(synchronized_message_traces.shape[1]/256)

        # Cut traces for indexes 2-254 and message bits 1-254
        index_traces = np.empty((synchronized_index_traces.shape[0], 253, index_seg_size+3*message_seg_size))
        message_traces = np.empty((synchronized_message_traces.shape[0], 254, message_seg_size))
        for t in range(synchronized_message_traces.shape[0]):
            for k in range(1,254):
                index_traces[t, k-1, :index_seg_size] = synchronized_index_traces[t, index_seg_size*k:index_seg_size*(k+1)]
                index_traces[t, k-1, index_seg_size:] = synchronized_message_traces[t, message_seg_size*(254-k):message_seg_size*(257-k)]

            for k in range(1,255):
                message_traces[t, k-1, :] = synchronized_message_traces[t, message_seg_size*k:message_seg_size*(k+1)]

        # Trim traces
        #index_traces, message_traces = trim_traces(index_traces, message_traces)

        # Save cut traces to array
        all_index_traces[CT_num] = index_traces
        all_message_traces[CT_num] = message_traces

        # Cut index 1 and 255 as well as message bits 0 and 255
        first_index_traces[CT_num, :, :index_seg_size] = synchronized_index_traces[:, -index_seg_size:]
        first_index_traces[CT_num, :, index_seg_size:] = synchronized_message_traces[:, :message_seg_size*3]
        last_index_traces[CT_num, :, :index_seg_size] = synchronized_index_traces[:, :index_seg_size]
        last_index_traces[CT_num, :, index_seg_size:] = synchronized_message_traces[:, -message_seg_size*2:]
        first_message_traces[CT_num] = synchronized_message_traces[:, :message_seg_size]
        last_message_traces[CT_num] = synchronized_message_traces[:, -message_seg_size:]

    # Standardize all traces
    print("Standardizing traces")
    index_traces = standardize_traces(all_index_traces)
    message_traces = standardize_traces(all_message_traces)
    first_index_traces = standardize_traces(first_index_traces)
    last_index_traces = standardize_traces(last_index_traces)
    first_message_traces = standardize_traces(first_message_traces)
    last_message_traces = standardize_traces(last_message_traces)

    # Save all traces to a dictionary
    traces = {}
    traces["index_traces"] = index_traces
    traces["message_traces"] = message_traces
    traces["first_index_traces"] = first_index_traces
    traces["last_index_traces"] = last_index_traces
    traces["first_message_traces"] = first_message_traces
    traces["last_message_traces"] = last_message_traces

    return traces


def recover_message(set, models, traces, part, CT_num, k1, k0, use_traces):
    stacks = get_predictions(set, models, traces, part, CT_num, k1, k0, use_traces)
    message = np.where(np.sum(stacks, axis=0)<0, 0, 1)

    return message


def load_true_secret_key():
    sk_filename = truth_folder+"SecKey.npy"
    sk = np.load(sk_filename)

    return sk


def load_all_models():
    models = {}
    models["index_models"] = load_models(index_model_folder, "kyber_index_[0-9]*.h5", MAX_NUM_OF_MODELS)
    if BIT_IN_BYTE_MODELS:
        models["message_bit0_models"] = load_models(message_bit0_model_folder, "kyber_message_[0-9]*.h5", MAX_NUM_OF_MODELS)
        models["message_bit7_models"] = load_models(message_bit7_model_folder, "kyber_message_[0-9]*.h5", MAX_NUM_OF_MODELS)
    else:
        models["message_models"] = load_models(message_model_folder, "kyber_message_[0-9]*.h5", MAX_NUM_OF_MODELS)
    models["last_index_models"] = load_models(beginning_index_model_folder, "kyber_index_beginning_[0-9]*.h5", MAX_NUM_OF_MODELS)
    models["last_message_models"] = load_models(beginning_message_model_folder, "kyber_message_beginning_[0-9]*.h5", MAX_NUM_OF_MODELS)
    models["first_index_models"] = load_models(end_index_model_folder, "kyber_index_end_[0-9]*.h5", MAX_NUM_OF_MODELS)
    models["first_message_models"] = load_models(end_message_model_folder, "kyber_message_end_[0-9]*.h5", MAX_NUM_OF_MODELS)
    
    return models


def attack(use_traces, set=None):
    filenum = 0

    # Recover all the messages
    recovered_messages = np.zeros((3, len(ECC_tool.ct_table), 256), dtype=int)
    for part in range(3):

        if os.path.exists(f"predictions/set_{set}/Part{part}_CT{ECC_tool.ct_table[-1][0]}-{ECC_tool.ct_table[-1][1]}/"):
            models = None
            traces = None
        else:
            # Load models
            models = load_all_models()

            # Load and process traces
            traces = load_traces(filenum, part)

        for CT_num, (k1, k0, k2) in enumerate(ECC_tool.ct_table):
            print(f"Predicting message: part-{part}, k1-{k1}, k0-{k0}, k2-{k2}")
            recovered_messages[part, CT_num] = recover_message(set, models, traces, part, CT_num, k1, k0, use_traces)
            print()

        del traces

    # Predict the secret key from the recovered messages
    ECC_tool.predict_secret_key(recovered_messages)

    # Negate all coefficients that wrapped around in rotation
    for i in range(1,768,2):
        ECC_tool.secret_coefficients[i] *= -1

    # Read true secret key
    sk = load_true_secret_key()

    # Print the predicted key compared to the true secret key
    ECC_tool.compare_against_true_secret_key(sk)
    print()

    # If attacking sets, save the attack-statistics to a file
    if SAVE_STATISTICS_FILE and not set is None:
        f = open(SAVE_STATISTICS_FILE, "a")
        f.write(f"Code distance              : {CODE_DISTANCE}\n")
        f.write(f"Set                        : {set}\n")
        f.write(f"Repeated traces            : {use_traces}\n")
        f.write(f"Corrected single errors    : {ECC_tool.stats['single_errors_corrected']}\n")
        f.write(f"Corrected double errors    : {ECC_tool.stats['double_errors_corrected']}\n")
        f.write(f"Corrected triple errors    : {ECC_tool.stats['triple_errors_corrected']}\n")
        f.write(f"Detected failures          : {ECC_tool.stats['detected_failures']}\n")
        f.write(f"Undetected failures        : {ECC_tool.stats['undetected_failures']}\n")
        f.write("\n")
        f.close()


def main():
    global data_folder
    global truth_folder
    global CODE_DISTANCE
    global ECC_tool

    # ./attack.py <set>
    if len(sys.argv) == 2:
        set = int(sys.argv[1])
        for cd in range(8, 1, -1):
            print("attacking set", set)
            CODE_DISTANCE = cd
            ECC_tool = ECC_CCT_TOOL("kyber768", CODE_DISTANCE)
            data_folder = data_folder_prefix+f"set_{set}/"
            truth_folder = data_folder+"GNDTruth/"
            for use_traces in range(1, REPS+1):
                attack(use_traces, set)

start = time.time()
main()
end = time.time()

print(f"The total running time was: {((end-start)/60):.2f} minutes.") 
