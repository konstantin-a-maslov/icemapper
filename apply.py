import icemapper
import xarray
import rioxarray
import rasterio
import numpy as np
import pickle
import argparse
import gc
from tqdm import tqdm


def apply(grd_scenes, insar_scenes, model, patch_size, timesteps=15, n_features=2, n_classes=2, nd_tol=1e2):
    ref = grd_scenes[0]
    _, height, width = ref.data.shape
    
    probs = np.zeros((height, width, n_classes), dtype=np.float32)
    weights = np.zeros((height, width, 1), dtype=np.float32)
    weight_kernel = gaussian_kernel(patch_size)[..., np.newaxis].astype(np.float32)
    
    total = 0 # first, estimate the total amount of patches
    row = 0
    while row < height:
        col = 0
        while col < width:
            total += 1
            col += (patch_size // 2)
        row += (patch_size // 2)
    
    with tqdm(total=total, desc="Inference...") as pbar:
        row = 0
        while row < height:
            col = 0
            while col < width:
                patch = np.zeros((1, timesteps, patch_size, patch_size, n_features), dtype=np.float32)
                y_slice, x_slice = slice(row, min(row + patch_size, height)), slice(col, min(col + patch_size, width))
                for f_idx, f_arrs in enumerate([grd_scenes, insar_scenes]):
                    for t_idx, f_arr in enumerate(f_arrs):
                        f_patch = f_arr.data[0, y_slice, x_slice]
                        p_height, p_width = f_patch.shape
                        py_slice, px_slice = slice(0, p_height), slice(0, p_width)
                        patch[0, t_idx, py_slice, px_slice, f_idx] = f_patch
                        
                mask = (np.isnan(patch)) | (np.isinf(patch)) | (patch > nd_tol)
                patch[mask] = 0
                p_probs = model.predict(patch, verbose=0)[0].astype(np.float32)
                p_probs = p_probs[py_slice, px_slice, :]
                probs[y_slice, x_slice, :] += p_probs * weight_kernel[py_slice, px_slice, :]
                weights[y_slice, x_slice, :] += weight_kernel[py_slice, px_slice, :]
                        
                pbar.update(1)
                col += (patch_size // 2)
            
            del patch
            gc.collect()
            row += (patch_size // 2)
            
    probs /= weights
    return probs


def gaussian_kernel(size, mu=0, sigma=0.5):
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    distance = np.sqrt(x**2 + y**2)
    kernel = np.exp(-(distance - mu)**2 / 2 / sigma**2) / np.sqrt(2 / np.pi) / sigma
    return kernel


def main(args):
    timesteps = 15
    
    if args.grd_paths is None or args.insar_paths is None:
        raise ValueError("GRD paths and InSAR coh paths must be provided")
    if len(args.grd_paths) != timesteps or len(args.insar_paths) != timesteps:
        raise ValueError(f"{timesteps} GRD and {timesteps} InSAR coh scenes must be provided")
        
    with open("data_stats.pickle", "rb") as stats_src:
        mins, maxs = pickle.load(stats_src)
    
    grd_scenes = []
    insar_scenes = []
    
    ref = None
    if args.ref_path:
        ref = rioxarray.open_rasterio(args.ref_path)
    
    for path in tqdm(args.grd_paths, desc="Reading GRD data..."):
        scene = rioxarray.open_rasterio(path)
        scene = (scene - mins["grd"]) / (maxs["grd"] - mins["grd"])
        if ref is None:
            ref = scene
            grd_scenes.append(ref)
        else:
            scene_reproj = scene.rio.reproject_match(ref, resampling=rasterio.enums.Resampling.bilinear, nodata=0)
            del scene
            gc.collect()
            grd_scenes.append(scene_reproj)
    for path in tqdm(args.insar_paths, desc="Reading InSAR data..."):
        scene = rioxarray.open_rasterio(path)
        scene = (scene - mins["insar"]) / (maxs["insar"] - mins["insar"])
        scene_reproj = scene.rio.reproject_match(ref, resampling=rasterio.enums.Resampling.bilinear, nodata=0)
        del scene
        gc.collect()
        insar_scenes.append(scene_reproj)
    
    patch_size, features = 384, ["grd", "insar"]
    model_name = args.model_name
    model = icemapper.ICEmapper_v2((timesteps, patch_size, patch_size, len(features)), 2, name=model_name)
    model.load_weights(f"weights/{model_name}.h5")
    
    probs = apply(grd_scenes, insar_scenes, model, patch_size, timesteps=timesteps)
    probs = np.moveaxis(probs, -1, 0)
    
    output = xarray.DataArray(
        data=probs,
        dims=["band", "y", "x"],
        coords={"band": [0, 1], "y": ref.coords["y"], "x": ref.coords["x"]},
    )
    output.rio.write_crs(ref.rio.crs, inplace=True)
    output.rio.write_transform(ref.rio.transform(), inplace=True)
    output.rio.to_raster(args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path", help="Folder to save .tif probs")
    parser.add_argument("--grd_paths", nargs="*", help="GRD scene paths")
    parser.add_argument("--insar_paths", nargs="*", help="InSAR scene paths")
    parser.add_argument("--ref_path", help="Reference raster path for reprojection")
    parser.add_argument("--model_name", default="20241009_ICEmapper_v2_grdinsar", help="Model name")
    args = parser.parse_args()
    
    main(args)
 