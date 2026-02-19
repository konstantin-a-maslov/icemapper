import icemapper
import tensorflow as tf
import h5py
import pickle
import numpy as np
from tqdm import tqdm
import argparse


def apply(model, arr, patch_size, n_classes=2):
    _, height, width, _ = arr.shape
    weighted_probs = np.zeros((height, width, n_classes))
    weights = gaussian_kernel(patch_size)[..., np.newaxis]
    counts = np.zeros((height, width, 1))

    row = 0
    while row + patch_size <= height:
        col = 0 
        while col + patch_size <= width:
            patch = arr[np.newaxis, :, row:row + patch_size, col:col + patch_size, :]
            patch_probs = model.predict(patch, verbose=0)[0]
            weighted_probs[row:row + patch_size, col:col + patch_size, :] += (weights * patch_probs)
            counts[row:row + patch_size, col:col + patch_size, :] += weights
            col += (patch_size // 2)
        row += (patch_size // 2)
    
    probs = weighted_probs / counts
    return probs


def gaussian_kernel(size, mu=0, sigma=0.5):
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    distance = np.sqrt(x**2 + y**2)
    kernel = np.exp(-(distance - mu)**2 / 2 / sigma**2) / np.sqrt(2 / np.pi) / sigma
    return kernel


def evaluate_from_basic_metrics(tp, tn, fp, fn):
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn)
    iou = tp / (tp + fp + fn)
    evaluation = {
        "tp": tp, "tn":  tn, "fp": fp, "fn": fn,
        "accuracy": accuracy,
        "precision": precision, "recall": recall,
        "f1": f1, "iou": iou
    }
    return evaluation


def evaluate(ref, pred, target_class=1):
    tp = np.sum((ref == target_class) & (pred == target_class))
    tn = np.sum((ref != target_class) & (pred != target_class))
    fp = np.sum((ref != target_class) & (pred == target_class))
    fn = np.sum((ref == target_class) & (pred != target_class))
    return evaluate_from_basic_metrics(tp, tn, fp, fn)


def main(args):
    features = args.features
    model_name = args.model_name
    model_builder = icemapper.ICEmapper_v2 if args.version == "v2" else icemapper.ICEmapper
    
    data_folder = "/extravolume/data/icemapper"
    dataset_path = f"{data_folder}/test.hdf5"
    output_probs_path = f"outputs/{model_name}/probs.hdf5"
    output_eval_path = f"outputs/{model_name}/eval.pickle"
    
    timesteps = 15
    patch_size = 384
    
    model = model_builder((timesteps, patch_size, patch_size, len(features)), 2, name=model_name)
    model.load_weights(f"weights/{model_name}.h5")
    
    dataset = h5py.File(dataset_path, "r")
    probs_output = h5py.File(output_probs_path, "w")
    
    evaluation = {
        "per_tile": {},
    }
    
    for tile_name in tqdm(dataset.keys()):
        tile = dataset[tile_name]
        
        arr = []
        for feature in features:
            time_index = tile.attrs[f"{feature}_base_time_idx"]
            feature_arr = tile[feature][time_index, ...]
            arr.append(feature_arr)
        arr = np.concatenate(arr, axis=-1)
        
        probs = apply(model, arr, patch_size)
        ref = dataset[tile_name]["outlines"]
        
        orig_height, orig_width = tile.attrs["orig_height"], tile.attrs["orig_width"]
        pad_height, pad_width = tile.attrs["pad_height"], tile.attrs["pad_width"]
        y_slice, x_slice = slice(pad_height, pad_height + orig_height), slice(pad_width, pad_width + orig_width)
        
        probs = probs[y_slice, x_slice, :]
        ref = ref[y_slice, x_slice, :]
        pred = np.argmax(probs, axis=-1)
        ref = np.argmax(ref, axis=-1)
        evaluation["per_tile"][tile_name] = evaluate(ref, pred)
        
        gpr = probs_output.create_group(tile_name)
        gpr.create_dataset("probs", data=probs, dtype=np.float32)
    
    dataset.close()
    probs_output.close()
        
    tp, tn, fp, fn = 0, 0, 0, 0
    for tile_name in evaluation["per_tile"].keys():
        tp += evaluation["per_tile"][tile_name]["tp"]
        tn += evaluation["per_tile"][tile_name]["tn"]
        fp += evaluation["per_tile"][tile_name]["fp"]
        fn += evaluation["per_tile"][tile_name]["fn"]
    evaluation["overall"] = evaluate_from_basic_metrics(tp, tn, fp, fn)
    print(model_name, evaluation["overall"])
    
    with open(output_eval_path, "wb") as eval_dst:
        pickle.dump(evaluation, eval_dst)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--model_name", default="20241009_ICEmapper_v2_grdinsar", help="Model name")
    parser.add_argument("-f", "--features", default=["grd", "insar"], nargs="*", help="Input features")
    parser.add_argument("-v", "--version", default="v2", choices=["v1", "v2"], help="ICEmapper version")
    args = parser.parse_args()
    
    main(args)
