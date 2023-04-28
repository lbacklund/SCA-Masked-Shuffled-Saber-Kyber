import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys
import fnmatch
from scipy import stats

TRACE_FOLDER = "traces/profiling/"

def hamming_weight(n):
    return bin(np.uint8(n)).count("1")

def load_traces(file_number, plot=False):
    #Import training traces and labels

    shuffle_traces_all = np.load(TRACE_FOLDER+'shuffle_traces_synchronized_'+str(file_number)+'.npy', mmap_mode="r")
    message_traces_all = np.load(TRACE_FOLDER+'message_traces_synchronized_'+str(file_number)+'.npy', mmap_mode="r")
    shuffle_labels_all = np.load(TRACE_FOLDER+'shuffle_labels_synchronized_'+str(file_number)+'.npy').astype(int)
    message_labels_all = np.load(TRACE_FOLDER+'message_labels_synchronized_'+str(file_number)+'.npy').astype(int)

    print("shape of original shuffle traces",shuffle_traces_all.shape)
    print("shape of original message traces",message_traces_all.shape)
    print("shape of original shuffle labels",shuffle_labels_all.shape)
    print("shape of original message labels",message_labels_all.shape)

    shuffle_seg_size = int(shuffle_traces_all.shape[1]/255)
    message_offset = 0
    message_seg_size = int(message_traces_all.shape[1]/256)

    traces = np.empty((shuffle_labels_all.shape[0], shuffle_seg_size+2*message_seg_size))
    shuffle_labels = np.empty((shuffle_labels_all.shape[0],))
    message_labels = np.empty((message_labels_all.shape[0],))

    #trace_avg = traces.mean(axis=0) # mean of columns
    #plt.plot(trace_avg) 

    #labels_hw = np.vectorize(hamming_weight)(labels)
    #print("shape of labels hw",labels_hw.shape)
    #trace_0 = traces[np.where(labels_hw < 4)]
    #trace_1 = traces[np.where(labels_hw > 4)]
    #t_test(trace_0,trace_1)
    #plt.show()

    traces_per_byte = int(shuffle_labels_all.shape[0])

    labels_bits = np.empty((shuffle_labels_all.shape[0], 256), dtype=int)

    for n in range(shuffle_labels_all.shape[0]):
        for byte in range(32):
            for bit in range(8):
                labels_bits[n, byte*8+bit] = message_labels_all[n, byte] >> bit & 1
        labels_bits[n] = labels_bits[n][shuffle_labels_all[n]]

    for k in range(1):
        traces[:, :shuffle_seg_size] = shuffle_traces_all[:, shuffle_seg_size*k:shuffle_seg_size*(k+1)]
        traces[:, shuffle_seg_size:] = message_traces_all[:, message_seg_size*(254-k):message_seg_size*(256-k)]

        shuffle_labels[:] = shuffle_labels_all[:,255-k]
        message_labels[:] = labels_bits[:, 255-k]

        #labels_hw = np.vectorize(hamming_weight)(labels)
        #print("shape of labels hw",labels_hw.shape)
        #trace_0 = traces[np.where(labels_hw < 4)]
        #trace_1 = traces[np.where(labels_hw > 4)]
        #t_test(trace_0,trace_1)
        #plt.show()


    print("Shape of all traces", traces.shape)    
    print("Shape of all shuffle labels", shuffle_labels.shape)
    print("Shape of all message labels", message_labels.shape)

    if plot:
        for trace in traces[::8*(shuffle_labels_all.shape[0]+1)]:
            plt.plot(trace)
        plt.show()

        # average of cut-and-joined traces
        #trace_all_avg = traces.mean(axis=0) # mean of columns
        #print("shape of avg traces all",trace_all_avg.shape)
        #plt.plot(trace_all_avg) 
        #plt.show()

        # t-test on cut-and-joined traces
        shuffle_labels_hw = np.vectorize(hamming_weight)(shuffle_labels)
        print("shape of labels hw",shuffle_labels_hw.shape)
        trace_0 = traces[np.where(shuffle_labels_hw < 4)]
        trace_1 = traces[np.where(shuffle_labels_hw > 4)]
        (stat, pvalue) = stats.ttest_ind(trace_0, trace_1, equal_var=False)
        plt.plot(stat)
        plt.show()
        
    return (traces, shuffle_labels.astype('int'), message_labels.astype('int'))

def main():
    if len(sys.argv) >= 2:
        file_number = int(sys.argv[1])
        print("file_number =", file_number)
        traces, shuffle_labels, message_labels = load_traces(file_number, plot=True)
        np.save(TRACE_FOLDER+"cut_traces_beginning_"+str(file_number), traces)
        np.save(TRACE_FOLDER+"cut_shuffle_labels_beginning_"+str(file_number), shuffle_labels)
        np.save(TRACE_FOLDER+"cut_message_labels_beginning_"+str(file_number), message_labels)
    else:
        for file_name in os.listdir(TRACE_FOLDER):
            if fnmatch.fnmatch(file_name, "shuffle_traces_synchronized_[0-9]*.npy"):
                file_number = int(file_name[28:-4])
                print("file_number =", file_number)
                traces, shuffle_labels, message_labels = load_traces(file_number, plot=True)
                np.save(TRACE_FOLDER+"cut_traces_beginning_"+str(file_number), traces)
                np.save(TRACE_FOLDER+"cut_shuffle_labels_beginning_"+str(file_number), shuffle_labels)
                np.save(TRACE_FOLDER+"cut_message_labels_beginning_"+str(file_number), message_labels)

main()
