import numpy as np
import abc


class Sampler(abc.ABC):
    def __init__(self, dataset, patch_size, features, labels):
        self.dataset = dataset
        self.tiles = list(dataset.keys())
        self.patch_size = patch_size
        self.features = features
        self.labels = labels

    def set_dataloader(self, dataloader):
        self.dataloader = dataloader

    def access_tile_group(self, tile):
        group = self.dataset[tile]
        return group

    def index(self):
        self.n_patches = 0 
        for tile in self.tiles:
            attrs = self.dataset[tile].attrs
            height, width = attrs["height"], attrs["width"]
            self.n_patches += (height // self.patch_size) * (width // self.patch_size)

    def sample(self, before_sampling_plugins=None):
        self.sample_image()
        self.load_time_index()
        if before_sampling_plugins is not None:
            for plugin in before_sampling_plugins:
                plugin.before_sampling()
        self.sample_patch()
        return self.patch
            
    def reset(self):
        pass

    @abc.abstractmethod
    def sample_image(self):
        raise NotImplementedError
        
    def load_time_index(self):
        self.time_index = {
            f: self.tile_group.attrs[f"{f}_base_time_idx"].copy() for f in self.features 
        }
        
    @abc.abstractmethod
    def sample_patch(self):
        raise NotImplementedError
        
    def _extract_patch(self):
        stack = []
        for feature in self.features:
            feature_patch = self.tile_group[feature][
                self.time_index[feature], self.y:self.y + self.patch_size, self.x:self.x + self.patch_size, :
            ]
            stack.append(feature_patch)
        self.patch = {}
        self.patch["inputs"] = np.concatenate(stack, axis=-1)
        self.patch[self.labels] = self.tile_group[self.labels][
            self.y:self.y + self.patch_size, self.x:self.x + self.patch_size, :
        ].astype(np.float32)


class RandomSampler(Sampler):
    def __init__(self, dataset, patch_size, features, labels="outlines"):
        super(RandomSampler, self).__init__(dataset, patch_size, features, labels)

    def sample_image(self):
        tile = np.random.choice(self.tiles)
        self.tile_group = self.access_tile_group(tile)

    def sample_patch(self):
        height, width = self.tile_group.attrs["height"], self.tile_group.attrs["width"]
        if height == self.patch_size:
            self.y = 0
        else:
            self.y = np.random.choice(height - self.patch_size)
        if width == self.patch_size:
            self.x = 0
        else:
            self.x = np.random.choice(width - self.patch_size)
        self._extract_patch()


class ConsecutiveSampler(Sampler):
    def __init__(self, dataset, patch_size, features, labels="outlines"):
        super(ConsecutiveSampler, self).__init__(dataset, patch_size, features, labels)
        self.reset()

    def reset(self):
        self.tile_idx = 0
        self.x = 0
        self.y = 0

    def sample_image(self):
        if self.tile_idx >= len(self.tiles):
            self.reset()
        tile = self.tiles[self.tile_idx]
        self.tile_group = self.access_tile_group(tile)
        
    def sample_patch(self):
        height, width = self.tile_group.attrs["height"], self.tile_group.attrs["width"]
        if self.x + self.patch_size > width:
            self.x = 0
            self.y += self.patch_size
        if self.y + self.patch_size > height:
            self.y = 0
            self.x = 0
            self.tile_idx += 1
            self.sample_image()
        self._extract_patch()
        self.x += self.patch_size


class RAMTileGroup:
    def __init__(self, tile_group, keys=None, dtype=np.float32):
        self.datasets = {
            _: np.array(tile_group[_], dtype=dtype) 
            for _ in (tile_group.keys() 
            if keys is None else keys)
        }
        self.attrs = {
            _: tile_group.attrs[_] 
            for _ in tile_group.attrs.keys()
        }

    def __getitem__(self, key):
        return self.datasets[key].copy()


def move_sampler_to_ram(sampler, keys=None):
    import types

    sampler.cached_tile_groups = {}
    sampler.nonram_access_tile_group = sampler.access_tile_group

    def ram_access_tile_group(self, tile):
        if tile not in self.cached_tile_groups:
            tile_group = self.nonram_access_tile_group(tile)
            ram_tile_group = RAMTileGroup(tile_group, keys=keys)
            self.cached_tile_groups[tile] = ram_tile_group
        return self.cached_tile_groups[tile]

    sampler.access_tile_group = types.MethodType(ram_access_tile_group, sampler)
    return sampler
