import tensorflow as tf
import numpy as np
import dataloaders
import dataloaders.transformations
import joblib
import gc


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
            
        def sample_one_patch():
            sample = self.sampler.sample(before_sampling_plugins=self.before_sampling_plugins)
            for plugin in self.after_sampling_plugins:
                sample = plugin.after_sampling(sample)
            return sample
        
        self.batch_list = joblib.Parallel(n_jobs=self.batch_size, prefer="threads")(
            joblib.delayed(sample_one_patch)() 
            for _ in range(self.batch_size)
        )
        # self.batch_list = []
        # for _ in range(self.batch_size):
        #     sample = self.sampler.sample(before_sampling_plugins=self.before_sampling_plugins)
        #     for plugin in self.after_sampling_plugins:
        #         sample = plugin.after_sampling(sample)
        #     self.batch_list.append(sample)
        
        batch_x, batch_y = self.__reformat(self.batch_list)
        for plugin in self.on_finalising_plugins:
            batch_x, batch_y = plugin.on_finalising(batch_x, batch_y)
        return batch_x, batch_y
        
    def __reformat(self, batch_list):
        features = batch_list[0].keys()
        batch = {}
        for feature in features:
            item_shape = batch_list[0][feature].shape
            feature_arr = np.empty((self.batch_size, *item_shape), dtype=np.float32)
            for item_idx, item in enumerate(batch_list):
                feature_arr[item_idx] = item[feature]
            batch[feature] = feature_arr
        batch_x = {_: batch[_] for _ in batch if _ != self.labels}
        batch_y = batch[self.labels]
        return batch_x, batch_y

    
def DataLoader(
    dataset_path, 
    mode,
    patch_size, 
    features,
    batch_size, 
    labels="outlines", 
    prefetch=1, 
    timesteps=15, 
    n_classes=2, 
    len_factor=1
):
    if not mode in {"train", "val"}:
        raise ValueError
        
    if mode == "train":
        sampler = dataloaders.RandomSampler(dataset_path, patch_size, features)
    else:
        sampler = dataloaders.ConsecutiveSampler(dataset_path, patch_size, features)
    data_seq = DataSequence(sampler, [], batch_size, labels, len_factor)
    steps_per_epoch = len(data_seq)
    
    del sampler
    del data_seq
    
    def gen(
        dataset_path, 
        mode,
        patch_size, 
        features,
        batch_size,  
        labels,  
        len_factor
    ):
        dataset_path = dataset_path.decode()
        mode = mode.decode()
        features = [_.decode() for _ in features]
        labels = labels.decode()

        if mode == "train":
            sampler = dataloaders.RandomSampler(dataset_path, patch_size, features)
            plugins = [
                dataloaders.ShuffleTimeIndex(p=0.25),
                dataloaders.Augmentation([
                    dataloaders.transformations.random_vertical_flip(p=0.5),
                    dataloaders.transformations.random_horizontal_flip(p=0.5),
                    dataloaders.transformations.random_rotation(p=0.75),
                    dataloaders.transformations.crop_and_scale(patch_size=patch_size, scale=0.8, p=0.25),
                    dataloaders.transformations.add_gamma_correction(gamma_l=0.8, gamma_r=1.2, p=0.25),
                    dataloaders.transformations.add_gaussian_shift(sigma=0.025, p=0.25), 
                    # dataloaders.transformations.swap_timesteps(p=0.1),
                    # dataloaders.transformations.feature_occlusion(max_obstruction_size=patch_size // 2, p=0.5),
                    # dataloaders.transformations.reverse_time(p=0.25),
                ]),
                dataloaders.AddLabelSmoothing(),
            ]
        else:
            sampler = dataloaders.ConsecutiveSampler(dataset_path, patch_size, features)
            plugins = []
        sampler = dataloaders.move_sampler_to_ram(sampler, keys=features + [labels])
        data_seq = DataSequence(sampler, plugins, batch_size, labels, len_factor)
        steps_per_epoch = len(data_seq)
        
        while True:
            for step in range(steps_per_epoch):
                x, y = data_seq[step]
                yield x, y
            if mode == "val":
                break
        
        del sampler
        del data_seq
        gc.collect()
                
    n_features = len(features)
    dataloader = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
              {"inputs": tf.TensorSpec(shape=(batch_size, timesteps, patch_size, patch_size, n_features), dtype=tf.float32, name="inputs")},
              tf.TensorSpec(shape=(batch_size, patch_size, patch_size, n_classes), dtype=tf.float32)
        ),
        args=(dataset_path, mode, patch_size, features, batch_size, labels, len_factor)
    )
    if mode == "val":
        dataloader = dataloader.cache()
    if prefetch > 0:
        dataloader = dataloader.prefetch(prefetch)
    return dataloader, steps_per_epoch
