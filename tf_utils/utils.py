import subprocess
import tensorflow as tf
import numpy as np

def gpu_selection():
    def get_gpu_utilization():
        cmd = "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader"
        utilization = subprocess.check_output(cmd, shell=True)
        utilization = utilization.decode("utf-8").strip().split("\n")
        utilization = [int(x.replace(" %", "")) for x in utilization]
        return utilization
    
    gpu_usage = np.array(get_gpu_utilization())
    less_used = gpu_usage.argmin()
    # print(f'The gpu less used in this moment is n: {less_used} with an usage of {gpu_usage[less_used]}')
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices(gpus[less_used], 'GPU')
            # logical_gpus = tf.config.list_logical_devices('GPU')
            # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)