import tensorflow

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))