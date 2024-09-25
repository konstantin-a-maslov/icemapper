import abc
import dataloaders.transformations
import numpy as np


class Plugin(abc.ABC):
    def set_dataloader(self, dataloader):
        self.dataloader = dataloader

    def get_sampler(self):
        if not self.dataloader:
            raise ValueError()
        return self.dataloader.sampler

    def __init_subclass__(cls, **kwargs):
        cls.has_before_sampling_behaviour = not (cls.before_sampling == Plugin.before_sampling)
        cls.has_after_sampling_behaviour = not (cls.after_sampling == Plugin.after_sampling)
        cls.has_on_finalising_behaviour = not (cls.on_finalising == Plugin.on_finalising)

    def before_sampling(self):
        pass
    
    def after_sampling(self, sample):
        return sample

    def on_finalising(self, batch_x, batch_y):
        return batch_x, batch_y


class ShuffleTimeIndex(Plugin):
    def __init__(self, p):
        self.p = p
        
    def before_sampling(self):
        sampler = self.get_sampler()
        features = sampler.features
        time_index = sampler.time_index
        tile_group = sampler.tile_group
        for feature in features:
            t_min = 0
            t_max = tile_group[feature].shape[0] - 1
            new_index = []
            for t in time_index[feature]:
                new_t = t
                if np.random.rand() < self.p:
                    new_t += np.random.choice([-1, 1])
                    if new_t < t_min:
                        new_t = t_min
                    elif new_t > t_max:
                        new_t = t_max
                new_index.append(new_t)
            time_index[feature] = new_index


class Augmentation(Plugin):
    def __init__(self, transformations):
        self.transformations = transformations

    def after_sampling(self, sample):
        dataloaders.transformations.apply_transformations(
            sample, self.transformations
        )
        return sample


class AddLabelSmoothing(Plugin):
    def __init__(self, smoothing=0.1):
        self.smoothing = smoothing

    def on_finalising(self, batch_x, batch_y):
        n_classes = batch_y.shape[-1]
        batch_y = batch_y * (1 - self.smoothing) + self.smoothing / n_classes
        return batch_x, batch_y
