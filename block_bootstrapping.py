import rioxarray
import geopandas
import numpy as np
import joblib
import pickle
import datetime


def shannon_entropy(x, eps=1e-6, C=2):
    x = np.clip(x, eps, 1 - eps)
    return -(1 - x) * np.log(1 - x) / np.log(C) - x * np.log(x) / np.log(C)


def shannon_confidence(x):
    return 1 - shannon_entropy(x)


def pixels_n_to_km_squared(pixels_n):
    return pixels_n * 10 * 10 / 1e6


def read_probs(year, eps=1e-8):
    print(f"{datetime.datetime.now()} Reading {year} data")
    probs = rioxarray.open_rasterio(f"/extravolume_global/icemapper/raster/results_202424d/postprocessed/temporally_filtered/{year}_probs_with2024.tif")[0]
    mask = (probs > eps)
    return probs, mask


def predict_subset(model, subset):
    return model.predict(subset)


def apply_model_multicore(model, data, n_jobs=32, chunk_size=16384):
    chunks = [data[i:i + chunk_size] for i in range(0, data.shape[0], chunk_size)]
    results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(predict_subset)(model, chunk) for chunk in chunks
    )
    return np.concatenate(results)


def get_confs(probs, mask, year):
    print(f"{datetime.datetime.now()} Calculating {year} confidence")
    probs_flatten = probs.data.flatten()
    mask_flatten = mask.data.flatten()
    probs_flatten_masked = probs_flatten[mask_flatten]
    confs_flatten_masked = shannon_confidence(probs_flatten_masked)[:, np.newaxis]
    print(f"{datetime.datetime.now()} Calibrating {year} confidence")
    with open("confidence_calibration_models/confidence_calibration_model_ICEmapper_v2_grdinsar.pickle", "rb") as mdl_src:
        calib_model = pickle.load(mdl_src)
    confs_flatten_masked = apply_model_multicore(calib_model, confs_flatten_masked)
    confs_flatten_masked[confs_flatten_masked < 0] = 0
    confs_flatten_masked[confs_flatten_masked > 1] = 1
    confs_flatten = probs_flatten.copy()
    confs_flatten[mask_flatten] = confs_flatten_masked
    confs_flatten[~mask_flatten] = 1.0
    confs = confs_flatten.reshape(probs.shape)
    return confs


def calculate_tile_area(probs, tile_geom):
    import rioxarray
    tile_probs = probs.rio.clip_box(*tile_geom.bounds)
    tile_area = pixels_n_to_km_squared(np.sum(tile_probs))
    # tile_area = pixels_n_to_km_squared(np.sum(tile_probs > 0.5))
    return tile_area


def block_bootstrapping_iter(areas, seed):
    rng = np.random.default_rng(seed)
    n_samples = len(areas)
    samples = rng.choice(areas, size=n_samples, replace=True)
    area = np.sum(samples)
    return area

    
def infer_for_a_year(year, tile_path, n_runs=10000, n_jobs=96, seed=None):
    probs, mask = read_probs(year)
    confs = get_confs(probs, mask, year)
    calib_probs = probs.copy()
    calib_probs.data = confs.copy()
    not_ice_mask = probs.data < 0.5
    calib_probs.data[not_ice_mask] = (1 - calib_probs.data[not_ice_mask])
    calib_probs.data[~mask.data] = 0
    if seed is None:
        seed = np.random.randint(0, 100000)
    print(f"{datetime.datetime.now()} Running block bootstrapping for {year}, seed={seed}")
    tiles = geopandas.read_file(tile_path)
    aoi = geopandas.read_file("/extravolume/data/icemapper/vector/aoi_utm33n.shp")
    tiles = tiles[tiles.geometry.intersects(aoi.iloc[0].geometry)]
    tile_areas = joblib.Parallel(n_jobs=n_jobs, verbose=51)(
        joblib.delayed(calculate_tile_area)(calib_probs, tile_geom) 
        for tile_geom in tiles.geometry
    )
    tile_areas = np.array(tile_areas)
    areas = joblib.Parallel(n_jobs=n_jobs, verbose=51)(
        joblib.delayed(block_bootstrapping_iter)(tile_areas, seed * (seed_salt + 1)) 
        for seed_salt in range(n_runs)
    )
    return areas


def main():
    all_years = ["2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024"]
    for year in all_years:
        areas = infer_for_a_year(year, "bootstrapping_grids/3.2km.shp", seed=42)
        with open(f"block_boostrapping_results_{year}_3.2km_20250116.pickle", "wb") as dst:
            pickle.dump(areas, dst)


if __name__ == "__main__":
    main()
    