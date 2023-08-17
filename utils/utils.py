import torch
import matplotlib.pyplot as plt
import csv
import nvidia_smi


def get_least_utilized_gpu():
    """
    Check which GPUs are available and returns the GPU with less memory occupied
    :return: torch.device object with the least utilized GPU
    """

    devices = []
    nvidia_smi.nvmlInit()
    deviceCount = nvidia_smi.nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        gpu_util = (nvidia_smi.nvmlDeviceGetUtilizationRates(handle).gpu/100.0)
        mem_free = nvidia_smi.nvmlDeviceGetMemoryInfo(handle).free
        devices.append((gpu_util, mem_free, i))

    # Sort the list of GPUs by max memory free in decreasing order and if memory free is same, then by GPU utilization
    devices.sort(key=lambda x: (x[1], x[0]), reverse=True)
    dev = devices[0][2]
    print(f"GPU with least memory usage: {dev}")
    device = torch.device(f"cuda:{dev}")
    print(device)
    return device

def plot_and_save_graph(epochs, model_name, train_losses, val_losses):
    # Create the plot
    plt.figure(figsize=(10, 6))

    e = [x for x in range(1, epochs+1)]

    plt.plot(e, train_losses, label='Training Loss')
    plt.plot(e, val_losses, label='Validation Loss')

    # Add labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(model_name)
    plt.legend()

    # Save the plot to a file
    plt.savefig(f'./plots/{model_name}.png')

    # Display the plot
    plt.show()


def save_results(model_name, train_losses, val_losses, elapsed_time):
    training_info = {
        'model-name': model_name,
        'train-loss': train_losses[-1],
        'valid-loss': val_losses[-1],
        'elapsed-time': elapsed_time,
    }

    csv_filename = 'training_info.csv'

    with open(csv_filename, 'a', newline='') as csvfile:
        fieldnames = training_info.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # If the file is empty, write the header row
        if csvfile.tell() == 0:
            writer.writeheader()

        # Write the training information to the CSV file
        writer.writerow(training_info)
