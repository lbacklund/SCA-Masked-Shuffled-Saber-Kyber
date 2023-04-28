import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import fnmatch
import sys

TRACE_FOLDER = "traces/profiling/"

def standardize_traces(filenumber):
    print("\nStandardizing file", filenumber)
    print("Loading traces")
    traces = np.load(TRACE_FOLDER+f"cut_traces_{filenumber}.npy")

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
    np.save(TRACE_FOLDER+f"cut_traces_{filenumber}.npy", traces)

def main():
    if len(sys.argv) >= 2:
        file_number = int(sys.argv[1])
        print("file_number =", file_number)
        standardize_traces(file_number)
    else:
        file_numbers = []
        for file_name in os.listdir(TRACE_FOLDER):
            if fnmatch.fnmatch(file_name, "cut_traces_[0-9]*.npy"):
                file_numbers.append(int(file_name[11:-4]))
        file_numbers.sort()
        print("File_numbers =", file_numbers)
        
        for filenumber in file_numbers:
            standardize_traces(filenumber)

main()
