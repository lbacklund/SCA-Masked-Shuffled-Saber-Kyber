import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import fnmatch

TRACE_FOLDER = "traces/profiling/"

def standardize_traces():
    print("Standardizing")
    print("Loading traces")
    traces = np.load(TRACE_FOLDER+"cut_joined_traces_beginning.npy")

    print("Calculating means")
    means = np.mean(traces, axis=0)
    #print(means)
    #plt.plot(means)
    #plt.title("Means")
    #plt.show()

    print("Calculating standard deviations")
    stdevs = np.zeros((traces.shape[1],))
    for n in tqdm(range(traces.shape[1]), "Calculating stdevs"):
        stdevs[n] = np.std(traces[:, n])
    #print(stdevs)
    #plt.plot(stdevs)
    #plt.title("Standard deviation")
    #plt.show()

    #for i in range(5):
    #    tmp_traces = traces[500000*i:500000*i+100]
    #    for trace in tmp_traces:
    #        plt.plot(trace)
    #    plt.show()
    #    tmp_traces = (tmp_traces - means) / stdevs
    #    for trace in tmp_traces:
    #        plt.plot(trace)
    #    plt.show()


    print("Standardizing traces")
    traces = (traces - means) / stdevs
    
    print("Saving traces")
    np.save(TRACE_FOLDER+"cut_joined_traces_beginning.npy", traces)

def main():
    standardize_traces()

main()
