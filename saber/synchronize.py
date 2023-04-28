import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
import os
import fnmatch
import sys

SOURCE_TRACE_FOLDER = "traces/profiling/"
TARGET_TRACE_FOLDER = "traces/profiling/"
NUM_OF_TRACES = 5000
RESOLUTION = 32

def calculate_ttest(traces, labels, divisor):
    distributions = []
    statistics = []
    for index_num in tqdm(range(255,-1,-RESOLUTION), "Calculating tests"):
        traces_a = traces[np.where(labels[:,index_num] < divisor)]
        traces_b = traces[np.where(labels[:,index_num] > divisor)]
        distributions.append((len(traces_a), len(traces_b), index_num))
        (stat, pvalue) = stats.ttest_ind(traces_a, traces_b, equal_var=False)
        statistics.append((stat, "Index num: {}".format(index_num)))
    return statistics, distributions

def plot_all_individual(statistics, title):
    plt.clf()
    for i in tqdm(range(len(statistics)), "Plotting tests"):
        plt.plot(statistics[i][0], label=statistics[i][1])
    #plt.legend(loc="right")
    plt.xlabel("Sample")
    plt.ylabel("SOST value")
    plt.title(title)
    plt.show()

def hamming_weight(n):
    return bin(np.uint8(n)).count("1")

def print_distributions(distributions):
    for i in range(len(distributions)):
        print("Distribution for index num {}: ({}:{})".format(distributions[i][2], distributions[i][0], distributions[i][1]))

def synchronize_shuffle_traces(file_number):
    print("Synchronizing shuffle traces")
    traces = np.load(SOURCE_TRACE_FOLDER+"shuffle_traces_"+str(file_number)+".npy")[:NUM_OF_TRACES]
    labels = np.load(SOURCE_TRACE_FOLDER+"shuffle_labels_"+str(file_number)+".npy")[:NUM_OF_TRACES]
    
    #for trace in traces[:10]:
    #    plt.plot(trace)
    #plt.show()

    #plt.plot(np.mean(traces, axis=0))
    #plt.show()

    pattern = np.mean(traces[:, 300:450], axis=0) # Mean of all second spikes

    #plt.plot(pattern)
    #plt.show()

    correlations = [np.correlate(trace, pattern) for trace in tqdm(traces, "Correlating traces")]
    
    #for correlation in correlations[:100]:
    #    plt.plot(correlation)
    #plt.show()

    margin = len(pattern)
    seg_len = len(pattern)+2*margin
    new_traces = np.empty((traces.shape[0], seg_len*255))
    new_labels = np.empty((labels.shape[0], 256))
    c = 0
    for i in tqdm(range(traces.shape[0]), "Synchronizing traces"): # process every trace
        new_labels[c] = labels[i]
        trace = traces[i]
        # Shuffling
        correlation = correlations[i]
        peak_counter = 255
        threshold = 0.27
        shift = 0
        n = 270
        while n < traces.shape[1]-len(pattern): # Go through all points in shuffling
            if correlation[n] > threshold:
                peak_index = n + np.argmax(correlation[n:n+20])
                new_traces[c, seg_len*(255-peak_counter):seg_len*(256-peak_counter)] = trace[peak_index-margin+shift:peak_index+len(pattern)+margin+shift]
                n = peak_index + 140
                peak_counter -= 1
                if not peak_counter: break
            else:
                n += 1
        if peak_counter:
            print(str(peak_counter)+" shuffling peaks not found!")
            plt.plot(correlations[i])
            plt.axhline(y=threshold, color='red')
            plt.show()
        c += 1

    #for trace in new_traces[:10]:
    #    plt.plot(trace)
    #plt.show()
    
    #for trace in new_traces[:10]:
    #    byte_len = int(len(trace)/255)
    #    for index in range(0,255):
    #        plt.plot(trace[index*byte_len:(index+1)*byte_len])
    #plt.show()
   
    np.save(TARGET_TRACE_FOLDER+"shuffle_traces_synchronized_"+str(file_number), new_traces)
    np.save(TARGET_TRACE_FOLDER+"shuffle_labels_synchronized_"+str(file_number), new_labels)

    #labels = np.array([HW_keybyte_conversion(label) for label in new_labels])
    #statistics, distributions = calculate_ttest(new_traces, labels, 4)
    #print_distributions(distributions)
    #plot_all_individual(statistics, "T-test synchronized shuffle")

def synchronize_message_traces(file_number):
    print("Synchronizing message traces")
    traces = np.load(SOURCE_TRACE_FOLDER+"message_traces_"+str(file_number)+".npy")[:NUM_OF_TRACES]
    labels = np.load(SOURCE_TRACE_FOLDER+"message_labels_"+str(file_number)+".npy")[:NUM_OF_TRACES].astype('int')
    #labels = np.empty((NUM_OF_TRACES, 256))
    #for n in range(NUM_OF_TRACES):
    #    for byte in range(32):
    #        for bit in range(8):
    #            labels[n,byte*8+bit] = message_labels[n, byte] >> bit & 1
    #    labels[n] = labels[n][shuffle_labels[n]]

    #statistics, distributions = calculate_ttest(traces, labels, 0.5)
    #print_distributions(distributions)
    #plot_all_individual(statistics, "T-test message")

    #for trace in traces[:100]:
    #    plt.plot(trace)
    #plt.show()
 
    end_pattern = np.mean(traces[:, 52500:52900], axis=0)
    end_correlations = [np.correlate(trace, end_pattern) for trace in tqdm(traces, "Correlating end of traces")]
    #for correlation in end_correlations[:100]:
    #    plt.plot(correlation)
    #plt.show()

    bad_traces = []
    for i in range(NUM_OF_TRACES):
        if np.max(end_correlations[i][1000:52200]) > 1:
            bad_traces.append(i)
            print(i, "bad")
    print(len(bad_traces), "out of", NUM_OF_TRACES, "incorrectly cut")

    pattern = np.mean(traces[:, 310+115:520+115], axis=0) # Mean of all second spikes

    #plt.plot(pattern)
    #plt.show()

    correlations = [np.correlate(trace, pattern) for trace in tqdm(traces, "Correlating traces")]
    
    #for correlation in correlations[:100]:
    #    plt.plot(correlation)
    #plt.show()
    
    seg_len = 3*70
    new_traces = np.empty((traces.shape[0], seg_len*256))
    new_labels = np.empty((labels.shape[0], 32))
    c = 0
    
    for i in tqdm(range(traces.shape[0]), "Synchronizing traces"): # process every trace
        trace = traces[i]
        # Shuffling
        correlation = correlations[i]
        shift = 0
        threshold = 0.14
        n = 400
        peak_counter = 256
        while True:
            if correlation[n] > threshold:
                peak_index = n + np.argmax(correlation[n:n+80])
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
    
    #for trace in new_traces[:100]:
    #    plt.plot(trace)
    #plt.show()
    
    #for trace in new_traces[:5]:
    #    bit_len = int(len(trace)/256)
    #    for index in range(0,256):
    #        plt.plot(trace[index*bit_len:(index+1)*bit_len])
    #plt.show()
   
    np.save(TARGET_TRACE_FOLDER+"message_traces_synchronized_"+str(file_number), new_traces)
    np.save(TARGET_TRACE_FOLDER+"message_labels_synchronized_"+str(file_number), labels)

    #statistics, distributions = calculate_ttest(new_traces, labels, 4)
    #print_distributions(distributions)
    #plot_all_individual(statistics, "T-test synchronized message")

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
