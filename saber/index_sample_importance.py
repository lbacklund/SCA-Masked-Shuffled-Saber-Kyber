import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model, Model, Sequential
import os
import fnmatch
from tqdm import tqdm

##################################################
num_of_traces = 50000
threshold = 0.995

model_folder = "models/index/15k/"        
data_folder = "traces/profiling/"
model_name = "saber_index_*.h5"
################################################### 

def load_models(model_folder,model_file):
    models = []
    for file_name in os.listdir(model_folder):
        if fnmatch.fnmatch(file_name, model_file):
            model = load_model(model_folder+file_name)
            print("Loading:", file_name)
            #print(model.summary())
            models.append(model)
        else:
            continue
    print("Loaded a total of", len(models), "from", model_folder)
    return models

def load_traces(num_of_traces):
    #Import training traces and labels

    traces = np.load(data_folder+'cut_joined_traces.npy', mmap_mode="r")[15000*253:15000*253+num_of_traces, :]
    labels = np.load(data_folder+'cut_joined_shuffle_labels.npy', mmap_mode="r")[15000*253:15000*253+num_of_traces]

    print("Shape of all traces",traces.shape)    
    print("Shape of all labels",labels.shape)

    return traces, labels.astype('int')

def plot_input_weights(model, fig):
    fig.plot(model.layers[0].get_weights()[0])

def plot_accuracy_after_zeroing_point(model, fig, all_traces, true_label):
    bar_values = []

    predictions = model.predict(all_traces)
    predictions = np.argmax(predictions, axis=1)
    correct_num = np.sum(np.where(np.bitwise_xor(predictions, true_label)==0, 1, 0))
    original_accuracy = correct_num / len(true_label)

    for i in tqdm(range(model.input_shape[1]), "Analyzing sample points"):
        index_to_zero = np.zeros((len(true_label), model.input_shape[1]))
        index_to_zero[:, i] = 1
        predictions = model.predict(np.where(index_to_zero==0, all_traces, 0))
        predictions = np.argmax(predictions, axis=1)
        correct_num = np.sum(np.where(np.bitwise_xor(predictions, true_label)==0, 1, 0))
        bar_values.append(correct_num/len(true_label))

    x_axis = np.array([i for i in range(len(bar_values))])
    bar_values = np.array(bar_values)

    fig.bar(np.where(bar_values >= original_accuracy*threshold)[0], bar_values[np.where(bar_values >= original_accuracy*threshold)[0]], color="red")
    fig.bar(np.where(bar_values < original_accuracy*threshold)[0], bar_values[np.where(bar_values < original_accuracy*threshold)[0]], color="green")
    return bar_values / original_accuracy


models = load_models(model_folder, model_name)

all_traces, true_label = load_traces(num_of_traces)

sample_importance = np.empty((len(models), all_traces.shape[1]))

for i, model in enumerate(models):
    print("Model", i)

    plt.figure(figsize=(11.6, 6.5), dpi=165)

    # Plot input weights
    plt.subplot(2, 1, 1)
    plt.title("Input weights")
    plot_input_weights(model, plt)
    plt.xlabel("Sample")
    plt.ylabel("Weight")

    # Plot sample importance
    plt.subplot(2, 1, 2)
    plt.title("Zeroed sample accuracy")
    sample_importance[i] = plot_accuracy_after_zeroing_point(model, plt, all_traces, true_label)
    plt.xlabel("Sample")
    plt.ylabel("Accuracy")

    plt.suptitle(f"Sample importance for model {i}")
    
    #plt.show()

    # Save plots
    plt.savefig(f"sample_importance/index_sample_importance_{i}.png")

    # Clear subplots
    plt.subplot(2, 1, 1)
    plt.clf()
    plt.subplot(2, 1, 2)
    plt.clf()

np.save("sample_importance/index_sample_importance.npy", sample_importance)
