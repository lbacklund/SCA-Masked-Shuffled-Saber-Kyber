import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import fnmatch
import sys

SOURCE_TRACE_FOLDER = "traces/profiling/"
TARGET_TRACE_FOLDER = "traces/profiling/"
NUM_OF_TRACES = 5000
RESOLUTION = 32

def synchronize_shuffle_traces(file_number):
    print("Synchronizing shuffle traces")
    traces = np.load(SOURCE_TRACE_FOLDER+"shuffle_traces_"+str(file_number)+".npy")[:NUM_OF_TRACES]
    labels = np.load(SOURCE_TRACE_FOLDER+"shuffle_labels_"+str(file_number)+".npy")[:NUM_OF_TRACES]
    
    #for trace in traces[:10]:
    #    plt.plot(trace)
    #plt.show()

    #plt.plot(np.mean(traces, axis=0))
    #plt.show()

    pattern = np.mean(traces[:, 290:345-15], axis=0) # Mean of all second spikes

    #plt.plot(pattern)
    #plt.show()

    correlations = [np.correlate(trace, pattern) for trace in tqdm(traces, "Correlating traces")]
    
    #for correlation in correlations[:100]:
    #    plt.plot(correlation)
    #plt.show()

    margin = 45
    seg_len = 55+2*margin
    new_traces = np.empty((traces.shape[0], seg_len*255))
    new_labels = np.empty((labels.shape[0], 256))
    c = 0
    for i in tqdm(range(traces.shape[0]), "Synchronizing traces"): # process every trace
        new_labels[c] = labels[i]
        trace = traces[i]
        # Shuffling
        correlation = correlations[i]
        peak_counter = 255
        threshold = 0.05
        shift = 0
        n = 230
        while n < traces.shape[1]-len(pattern): # Go through all points in shuffling
            try:
                if correlation[n] > threshold:
                    peak_index = n + np.argmax(correlation[n:n+10])
                    new_traces[c, seg_len*(255-peak_counter):seg_len*(256-peak_counter)] = trace[peak_index-margin+shift:peak_index+seg_len-margin+shift]
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

    #for trace in new_traces[:10]:
    #    plt.plot(trace)
    #plt.show()
    
    for trace in new_traces[:10]:
        byte_len = int(len(trace)/255)
        for index in range(0,255):
            plt.plot(trace[index*byte_len:(index+1)*byte_len])
    plt.show()
   
    np.save(TARGET_TRACE_FOLDER+"shuffle_traces_synchronized_"+str(file_number), new_traces)
    np.save(TARGET_TRACE_FOLDER+"shuffle_labels_synchronized_"+str(file_number), new_labels)


def synchronize_message_traces(file_number):
    print("Synchronizing message traces")
    traces = np.load(SOURCE_TRACE_FOLDER+"message_traces_"+str(file_number)+".npy")[:NUM_OF_TRACES]
    labels = np.load(SOURCE_TRACE_FOLDER+"message_labels_"+str(file_number)+".npy")[:NUM_OF_TRACES].astype('int')

    #for trace in traces[:100]:
    #    plt.plot(trace)
    #plt.show()
 
    #plt.plot(np.mean(traces, axis=0))
    #plt.show()

    end_pattern = np.mean(traces[:, 58150:58400], axis=0)
    end_correlations = [np.correlate(trace, end_pattern) for trace in tqdm(traces, "Correlating end of traces")]
    #for correlation in end_correlations[:100]:
    #    plt.plot(correlation)
    #plt.show()

    bad_traces = []
    for i in range(NUM_OF_TRACES):
        if np.max(end_correlations[i][1000:58100]) > 0.5:
            bad_traces.append(i)
            #plt.plot(end_correlations[i])
            #plt.plot(traces[i])
            #plt.plot(end_correlations[0])
            #plt.show()
            print(i, "bad")
    print(len(bad_traces), "out of", NUM_OF_TRACES, "incorrectly cut")

    pattern = np.mean(traces[:, 900:1125-100], axis=0) # Mean of all second spikes

    #plt.plot(pattern)
    #plt.show()

    correlations = [np.correlate(trace, pattern) for trace in tqdm(traces, "Correlating traces")]
    
    #for correlation in correlations[:100]:
    #    plt.plot(correlation)
    #plt.show()
    
    seg_len = 225
    new_traces = np.empty((traces.shape[0], seg_len*256))
    new_labels = np.empty((labels.shape[0], 32))
    c = 0
    
    for i in tqdm(range(traces.shape[0]), "Synchronizing traces"): # process every trace
        trace = traces[i]
        # Shuffling
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
            #elif n > 500 and peak_counter == 256:
            #    print("Peak not found!")
            #    plt.plot(correlation)
            #    plt.show()
            else: n += 1
            if n >= len(correlation): break
        if peak_counter:
            print(str(peak_counter)+" message peaks not found!")
            plt.plot(correlations[i])
            plt.axhline(y=threshold, color='red')
            plt.show()
        c += 1
    
    #for trace in new_traces[:100]:
    #    plt.plot(trace)
    #plt.show()
    
    for trace in new_traces[:5]:
        bit_len = int(len(trace)/256)
        for index in range(0,256):
            plt.plot(trace[index*bit_len:(index+1)*bit_len])
    plt.show()
   
    np.save(TARGET_TRACE_FOLDER+"message_traces_synchronized_"+str(file_number), new_traces)
    np.save(TARGET_TRACE_FOLDER+"message_labels_synchronized_"+str(file_number), labels)


def main():
    if len(sys.argv) >= 2:
        file_number = int(sys.argv[1])
        print("file_number =", file_number)
        synchronize_message_traces(file_number)
        synchronize_shuffle_traces(file_number)
    else:
        for file_name in os.listdir(SOURCE_TRACE_FOLDER):
            if fnmatch.fnmatch(file_name, "shuffle_traces_[0-9]*.npy"):
                file_number = int(file_name[15:-4])
                print("file_number =", file_number)
                synchronize_message_traces(file_number)
                synchronize_shuffle_traces(file_number)

main()
