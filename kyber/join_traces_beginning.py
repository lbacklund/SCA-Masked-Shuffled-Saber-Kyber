import numpy as np
import os
import fnmatch
from tqdm import tqdm
from pathlib import Path

SOURCE_FOLDER = "traces/profiling/"
TARGET_FOLDER = "traces/profiling/"

def map_file_trim(params):
    (i, file_number, trace_folder, permutation, index_trim_mask, message_trim_mask, final_traces, final_shuffle_labels, final_message_labels) = params
    traces = np.load(SOURCE_FOLDER+"cut_traces_beginning_"+str(file_number)+".npy", mmap_mode="r")
    shuffle_labels = np.load(SOURCE_FOLDER+"cut_shuffle_labels_beginning_"+str(file_number)+".npy", mmap_mode="r")
    message_labels = np.load(SOURCE_FOLDER+"cut_message_labels_beginning_"+str(file_number)+".npy", mmap_mode="r")
    for n in tqdm(range(5000), "Mapping file "+str(file_number)):
        final_shuffle_traces[permutation[5000*i+n]] = traces[n, index_trim_mask]
        final_message_traces[permutation[5000*i+n]] = traces[n, message_trim_mask]
        final_shuffle_labels[permutation[5000*i+n]] = shuffle_labels[n]
        final_message_labels[permutation[5000*i+n]] = message_labels[n]

def join_trim(file_numbers, index_trim_mask, message_trim_mask):
    print("Creating new files")

    final_shuffle_traces = np.memmap(TARGET_FOLDER+"cut_joined_shuffle_traces_beginning.npy", shape=(5000*len(file_numbers), len(index_trim_mask)), offset=128, dtype='float64', mode="w+")
    header = np.lib.format.header_data_from_array_1_0(final_shuffle_traces)
    with open(TARGET_FOLDER+"cut_joined_shuffle_traces_beginning.npy", 'r+b') as f:
        np.lib.format.write_array_header_1_0(f, header)
    
    final_message_traces = np.memmap(TARGET_FOLDER+"cut_joined_message_traces_beginning.npy", shape=(5000*len(file_numbers), len(message_trim_mask)), offset=128, dtype='float64', mode="w+")
    header = np.lib.format.header_data_from_array_1_0(final_message_traces)
    with open(TARGET_FOLDER+"cut_joined_message_traces_beginning.npy", 'r+b') as f:
        np.lib.format.write_array_header_1_0(f, header)
    
    final_shuffle_labels = np.memmap(TARGET_FOLDER+"cut_joined_shuffle_labels_beginning.npy", shape=(5000*len(file_numbers),), offset=128, dtype='int', mode="w+")
    header = np.lib.format.header_data_from_array_1_0(final_shuffle_labels)
    with open(TARGET_FOLDER+"cut_joined_shuffle_labels_beginning.npy", 'r+b') as f:
        np.lib.format.write_array_header_1_0(f, header)

    final_message_labels = np.memmap(TARGET_FOLDER+"cut_joined_message_labels_beginning.npy", shape=(5000*len(file_numbers),), offset=128, dtype='int', mode="w+")
    header = np.lib.format.header_data_from_array_1_0(final_message_labels)
    with open(TARGET_FOLDER+"cut_joined_message_labels_beginning.npy", 'r+b') as f:
        np.lib.format.write_array_header_1_0(f, header)
    
    print("Creating permutation of length", len(final_shuffle_traces))
    permutation = np.random.permutation(len(final_shuffle_traces))
    np.save(TARGET_FOLDER+"join_permutation_beginning.npy", permutation)

    parameters = []
    for i, file_number in enumerate(file_numbers):
        parameters.append((i, file_number, SOURCE_FOLDER, permutation, index_trim_mask, message_trim_mask, final_traces, final_shuffle_labels, final_message_labels))
    for i in range(len(parameters)):
        map_file_trim(parameters[i])
    
    print("Done")

def map_file(params):
    (i, file_number, trace_folder, permutation, final_traces, final_shuffle_labels, final_message_labels) = params
    traces = np.load(SOURCE_FOLDER+"cut_traces_beginning_"+str(file_number)+".npy", mmap_mode="r")
    shuffle_labels = np.load(trace_folder+"cut_shuffle_labels_beginning_"+str(file_number)+".npy", mmap_mode="r")
    message_labels = np.load(trace_folder+"cut_message_labels_beginning_"+str(file_number)+".npy", mmap_mode="r")
    for n in tqdm(range(5000), "Mapping file "+str(file_number)):
        final_traces[permutation[5000*i+n]] = traces[n]
        final_shuffle_labels[permutation[5000*i+n]] = shuffle_labels[n]
        final_message_labels[permutation[5000*i+n]] = message_labels[n]

def join(file_numbers):
    print("Creating new files")

    trace_len = np.load(SOURCE_FOLDER+"cut_traces_beginning_0.npy", mmap_mode="r").shape[1]
    final_traces = np.memmap(TARGET_FOLDER+"cut_joined_traces_beginning.npy", shape=(5000*len(file_numbers), trace_len), offset=128, dtype='float64', mode="w+")
    header = np.lib.format.header_data_from_array_1_0(final_traces)
    with open(TARGET_FOLDER+"cut_joined_traces_beginning.npy", 'r+b') as f:
        np.lib.format.write_array_header_1_0(f, header)
    
    final_shuffle_labels = np.memmap(TARGET_FOLDER+"cut_joined_shuffle_labels_beginning.npy", shape=(5000*len(file_numbers),), offset=128, dtype='int', mode="w+")
    header = np.lib.format.header_data_from_array_1_0(final_shuffle_labels)
    with open(TARGET_FOLDER+"cut_joined_shuffle_labels_beginning.npy", 'r+b') as f:
        np.lib.format.write_array_header_1_0(f, header)

    final_message_labels = np.memmap(TARGET_FOLDER+"cut_joined_message_labels_beginning.npy", shape=(5000*len(file_numbers),), offset=128, dtype='int', mode="w+")
    header = np.lib.format.header_data_from_array_1_0(final_message_labels)
    with open(TARGET_FOLDER+"cut_joined_message_labels_beginning.npy", 'r+b') as f:
        np.lib.format.write_array_header_1_0(f, header)
    
    print("Creating permutation of length", len(final_traces))
    permutation = np.random.permutation(len(final_traces))
    np.save(TARGET_FOLDER+"join_permutation_beginning.npy", permutation)

    parameters = []
    for i, file_number in enumerate(file_numbers):
        parameters.append((i, file_number, SOURCE_FOLDER, permutation, final_traces, final_shuffle_labels, final_message_labels))
    for i in range(len(parameters)):
        map_file(parameters[i])

def main():
    file_numbers = []
    for file_name in os.listdir(SOURCE_FOLDER):
        if fnmatch.fnmatch(file_name, "cut_traces_beginning_[0-9]*.npy"):
            file_numbers.append(int(file_name[21:-4]))
    file_numbers.sort()
    print("File_numbers =", file_numbers)

    trim = False
    if Path("index_trim_mask_beginning.npy").exists():
        if Path("message_trim_mask_beginning.npy").exists():
            print("Loading trim masks")
            index_trim_mask = np.load("index_trim_mask_beginning.npy")
            message_trim_mask = np.load("message_trim_mask_beginning.npy")
            trim = True
        else:
            print("No trim masks, using all samples")
    else:
        print("No trim mask, using all samples")

    if trim:
        join_trim(file_numbers, index_trim_mask, message_trim_mask)
    else:
        join(file_numbers)

main()
