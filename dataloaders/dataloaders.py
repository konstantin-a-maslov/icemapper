import tensorflow as tf
import numpy as np


class DataSequence():
    def __init__(self, sampler, plugins, batch_size, labels="outlines", len_factor=1):
        self.sampler = sampler
        self.plugins = plugins
        self.batch_size = batch_size
        self.labels = labels
        self.len_factor = len_factor
        self.__index()

    def __index(self):
        self.sampler.set_dataloader(self)
        self.before_sampling_plugins = []
        self.after_sampling_plugins = []
        self.on_finalising_plugins = []
        for plugin in self.plugins:
            self.add_plugin(plugin)
        self.sampler.index()
    
    def __index_plugin(self, plugin):
        if plugin.has_before_sampling_behaviour:
            self.before_sampling_plugins.append(plugin)
        if plugin.has_after_sampling_behaviour:
            self.after_sampling_plugins.append(plugin)
        if plugin.has_on_finalising_behaviour:
            self.on_finalising_plugins.append(plugin)

    def add_plugin(self, plugin):
        plugin.set_dataloader(self)
        self.__index_plugin(plugin)

    def __len__(self):
        return self.sampler.n_patches // self.batch_size * self.len_factor

    def __getitem__(self, idx):
        if idx == 0:
            self.sampler.reset()
        self.batch_list = []
        self.sample_idx = 0
        while self.sample_idx < self.batch_size:
            sample = self.sampler.sample(before_sampling_plugins=self.before_sampling_plugins)
            for plugin in self.after_sampling_plugins:
                sample = plugin.after_sampling(sample)
            self.batch_list.append(sample)
            self.sample_idx += 1
        batch_x, batch_y = self.__reformat(self.batch_list)
        for plugin in self.on_finalising_plugins:
            batch_x, batch_y = plugin.on_finalising(batch_x, batch_y)
        return batch_x, batch_y
        
    def __reformat(self, batch_list):
        features = batch_list[0].keys()
        batch = {}
        for feature in features:
            feature_list = []
            for item in batch_list:
                feature_list.append(item[feature])
            batch[feature] = np.array(feature_list)
        batch_x = {_: batch[_] for _ in batch if _ != self.labels}
        batch_y = batch[self.labels]
        return batch_x, batch_y

    
def DataLoader(sampler, plugins, batch_size, labels="outlines", prefetch=4, timesteps=15, n_classes=2, len_factor=1):
    data_seq = DataSequence(sampler, plugins, batch_size, labels, len_factor)
    steps_per_epoch = len(data_seq)
    patch_size = sampler.patch_size
    n_features = len(sampler.features)
    def gen():
        while True:
            for step in range(steps_per_epoch):
                x, y = data_seq[step]
                yield x, y
    dataloader = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
              {"inputs": tf.TensorSpec(shape=(batch_size, timesteps, patch_size, patch_size, n_features), dtype=tf.float32, name="inputs")},
              tf.TensorSpec(shape=(batch_size, patch_size, patch_size, n_classes), dtype=tf.float32)
        )
    )
    dataloader = dataloader.prefetch(prefetch)
    return dataloader, steps_per_epoch
