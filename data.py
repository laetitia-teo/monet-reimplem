# import torch
# from tfrecord.torch.dataset import TFRecordDataset

# BATCH_SIZE = 32

# path = 'data/multi_dsprites_multi_dsprites_colored_on_colored.tfrecords'
# index_path = None
# description = {"image": "byte", "label": "float"}
# dataset = TFRecordDataset(path, index_path, description)
# loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)
import time
import torch
import tensorflow as tf
from multi_object_datasets import multi_dsprites

tf.enable_eager_execution()
BATCH_SIZE = 64

tf_records_path = 'data/multi_dsprites_multi_dsprites_colored_on_colored.tfrecords'
dataset = multi_dsprites.dataset(tf_records_path, 'colored_on_colored')
batched_dataset = dataset.batch(BATCH_SIZE)

def to_tensor(data):
    # for converting the output of the tf image data into a torch Tensor
    return torch.Tensor(data['image'].numpy()).permute(0, 3, 1, 2) / 255

if __name__ == '__main__':
    for data in batched_dataset:
        img = data['image']
        t = torch.Tensor(img.numpy()) / 255
        t1 = time.time()