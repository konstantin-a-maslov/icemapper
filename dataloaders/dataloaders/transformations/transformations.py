import numpy as np
import cv2


def random_vertical_flip(p=0.5):
    def transform(patch):
        if np.random.random() > p:
            return
        for feature in patch:
            original = patch[feature]
            expanded = False
            if len(original.shape) == 3:
                original = np.expand_dims(original, axis=0)
                expanded = True
            transformed = np.flip(original, axis=1)
            if expanded:
                transformed = transformed[0]
            patch[feature] = transformed
    return transform


def random_horizontal_flip(p=0.5):
    def transform(patch):
        if np.random.random() > p:
            return
        for feature in patch:
            original = patch[feature]
            expanded = False
            if len(original.shape) == 3:
                original = np.expand_dims(original, axis=0)
                expanded = True
            transformed = np.flip(original, axis=2)
            if expanded:
                transformed = transformed[0]
            patch[feature] = transformed
    return transform


def random_rotation(p=0.75):
    def transform(patch):
        if np.random.random() > p:
            return
        k = np.random.choice([1, 2, 3])
        for feature in patch:
            original = patch[feature]
            expanded = False
            if len(original.shape) == 3:
                original = np.expand_dims(original, axis=0)
                expanded = True
            transformed = np.rot90(original, k, axes=(2, 1))
            if expanded:
                transformed = transformed[0]
            patch[feature] = transformed
    return transform


def crop_and_scale(patch_size, scale=0.8, p=0.25):
    def transform(patch):
        if np.random.random() > p:
            return
        scale_coef = np.random.random() * (1 - scale) + scale
        new_size = int(scale_coef * patch_size)
        y = np.random.choice(patch_size - new_size)
        x = np.random.choice(patch_size - new_size)
        for feature in patch:
            original = patch[feature]
            expanded = False
            if len(original.shape) == 3:
                original = np.expand_dims(original, axis=0)
                expanded = True
            crop = original[:, y:y + new_size, x:x + new_size, :]
            timesteps, _, _, depth = crop.shape
            transformed = np.empty((timesteps, patch_size, patch_size, depth))
            for timestep in range(timesteps):
                for channel_idx in range(depth):
                    transformed[timestep, :, :, channel_idx] = cv2.resize(
                        crop[timestep, :, :, channel_idx], (patch_size, patch_size), 
                        interpolation=cv2.INTER_LINEAR
                    )
            if expanded:
                transformed = transformed[0]
            patch[feature] = transformed
    return transform


def add_gamma_correction(gamma_l, gamma_r, p=0.25):
    def transform(patch):
        if np.random.random() > p:
            return
        depth = patch["inputs"].shape[-1]
        gammas = np.random.rand(depth) * (gamma_r - gamma_l) + gamma_l
        patch["inputs"][...] = patch["inputs"] ** gammas
        patch["inputs"][np.isnan(patch["inputs"])] = 0
    return transform


def add_gaussian_shift(sigma, p=0.25):
    def transform(patch):
        if np.random.random() > p:
            return
        shape = (patch["inputs"].shape[-1], )
        shift = np.random.normal(scale=sigma, size=shape)
        patch["inputs"] += shift
    return transform


def apply_transformations(patch, transformations):
    if not transformations:
        return 
    for transformation in transformations:
        transformation(patch)
