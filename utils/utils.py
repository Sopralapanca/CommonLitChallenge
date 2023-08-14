import torch
import matplotlib.pyplot as plt
import csv


def get_least_utilized_gpu():
    """
    Check which GPUs are available and returns the GPU with less memory occupied
    :return:
    """
    device_count = torch.cuda.device_count()
    least_memory = float('inf')
    least_utilized_gpu = None

    for i in range(device_count):
        memory_allocated = torch.cuda.memory_allocated(device=i)
        if memory_allocated < least_memory:
            least_memory = memory_allocated
            least_utilized_gpu = i

    return least_utilized_gpu


# least_utilized_gpu = get_least_utilized_gpu()
# print(f"GPU with least usage: {least_utilized_gpu}")
# device = torch.device(f"cuda:{least_utilized_gpu}")

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
