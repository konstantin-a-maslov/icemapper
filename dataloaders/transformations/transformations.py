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
            timesteps, crop_height, crop_width, depth = crop.shape
            crop = np.reshape(crop, (crop_height, crop_width, timesteps * depth))
            transformed = cv2.resize(crop, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
            transformed = np.reshape(transformed, (timesteps, patch_size, patch_size, depth))
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


def add_gamma_correction_DEPR(gamma_l, gamma_r, p=0.25):
    def transform(patch):
        if np.random.random() > p:
            return
        timesteps, _, _, depth = patch["inputs"].shape
        gammas = np.random.rand(timesteps, 1, 1, depth) * (gamma_r - gamma_l) + gamma_l
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


def add_gaussian_shift_DEPR(sigma, p=0.25):
    def transform(patch):
        if np.random.random() > p:
            return
        timesteps, _, _, depth = patch["inputs"].shape
        shift = np.random.normal(scale=sigma, size=(timesteps, 1, 1, depth))
        patch["inputs"] += shift
    return transform


def swap_timesteps(p=0.1):
    def transform(patch):
        timesteps = patch["inputs"].shape[0]
        for timestep_idx in range(timesteps):
            if np.random.random() > p:
                continue
            other_idx = np.random.choice(timesteps)
            tmp = patch["inputs"][timestep_idx, ...].copy()
            patch["inputs"][timestep_idx, ...] = patch["inputs"][other_idx, ...]
            patch["inputs"][other_idx, ...] = tmp
    return transform


def feature_occlusion(max_obstruction_size=192, p=0.5):
    def transform(patch):
        _, height, width, _ = patch["inputs"].shape
        if np.random.random() > p:
            return
        obstruction_height = int(np.random.random() * max_obstruction_size)
        obstruction_width = int(np.random.random() * max_obstruction_size)
        y = np.random.choice(height - obstruction_height)
        x = np.random.choice(width - obstruction_width)
        patch["inputs"][:, y:y + obstruction_height, x:x + obstruction_width, :] = 0
    return transform


def reverse_time(p=0.25):
    def transform(patch):
        if np.random.random() > p:
            return
        patch["inputs"] = patch["inputs"][::-1, :, :, :]
    return transform


def apply_transformations(patch, transformations):
    if not transformations:
        return 
    for transformation in transformations:
        transformation(patch)
