# ICEmapper

[Konstantin A. Maslov](https://people.utwente.nl/k.a.maslov), [Thomas Schellenberger](https://www.mn.uio.no/geo/english/people/aca/geohyd/thosche/), [Claudio Persello](https://people.utwente.nl/c.persello), [Alfred Stein](https://people.utwente.nl/a.stein)

[[`Paper`](https://doi.org/10.31223/X5472T)] [[`Datasets`](#datasets)] [[`BibTeX`](#citing)] 

<br/>

![results](assets/results.png)

The rapid warming in polar regions highlights the need to monitor climate change impacts such as glacier retreat and related global sea level rise. 
Glacier area is an essential climate variable but its tracking is complicated by the labour-intensive manual digitisation of satellite imagery.
Here we introduce ICEmapper, a deep learning model that maps glacier outlines from Sentinel-1 time series with accuracy on par with human experts.
We used this model to retrieve Svalbard glacier outlines for 2016&ndash;2024 and found a tripling of the glacier area loss rate in the last decade (-227 km<sup>2</sup> a<sup>-1</sup>) as compared to that of 1970&ndash;2006 (-68 km<sup>2</sup> a<sup>-1</sup>). 
Our analysis shows significant area changes related to glacier surging, namely, the Nathorstbreen system and Austfonna, Basin-3 surges. 
These two surges collectively added to the area change in 2006&ndash;2016 (+194.30 km<sup>2</sup> or +0.60%), thus delaying the regionwide area loss by two&ndash;three years. 
In contrast, during 2016&ndash;2024, surging glaciers showed statistically significantly faster area loss rates than non-surging glaciers.
Our results indicate a significant acceleration in glacier area loss in Svalbard, and we anticipate broader applications of our method to track glacier changes on larger scales. 

<br/>

## Data access

The ICEmapper training dataset is publicly available at [DATASET-URL-HERE](DATASET-URL-HERE). 
Download and unpack the dataset into a directory of your choice. 
Adjust the folder paths in the dependent scripts (`train.py`, `predict_and_evaluate.py`, `confidence_calibration.ipynb` and `block_bootstrapping.py`) accordingly. 


## Getting started

### Repository layout

- `icemapper/`&mdash;model definitions (`ICEmapper` for max-pool and `ICEmapper_v2` for time-weighted pool, controlled by setting `-v` to `v1` or `v2` where applicable)
- `layers/`&mdash;model building blocks
- `dataloaders/`&mdash;HDF5 sampling, augmentation, and `tf.data` pipelines
- `utils/`&mdash;focal loss, IoU metric, LR scheduling callback
- `weights/`&mdash;pretrained model weights (`.h5`)
- `logs/`&mdash;training logs
- `rates_resampling/`&mdash;historical trend reevaluations
- `train.py`&mdash;training script
- `predict_and_evaluate.py`&mdash;tiled inference on an HDF5 split + evaluation
- `apply.py`&mdash;inference on user-provided GeoTIFF time series (GRD + InSAR coherence by default)
- `confidence_calibration.ipynb` + `confidence_calibration_models/`&mdash;confidence calibration workflow/models
- `block_bootstrapping.py`&mdash;block bootstrapping for area uncertainty estimation

> Tip: run scripts from the **repository root** (imports assume this working directory), except `rates_resampling`.


### Installation

We recommend using the [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) Python distributions. 
After installing one of them, one can use the `conda` package manager to install the required libraries in a new environment called `massive-tf` and activate it by running
```
conda create -n massive-tf "tensorflow<2.16" h5py scikit-learn rioxarray geopandas jupyterlab tqdm matplotlib -c conda-forge
conda activate massive-tf
```
We tested this configuration on Ubuntu 22.04 (see `env.yml` for the exported environment). 
We also expect it to work on any modern Linux distribution or Windows, given properly configured NVIDIA GPU drivers.


### Training

Training is configured in `train.py`:
- `data_folder`&mdash;path to your dataset directory
- `model_name`
- `features`&mdash;feature set, default is `["grd", "insar"]`, combinations of `grd`, `rtc` and `insar` are allowed
- `model`&mdash;`icemapper.ICEmapper(...)` for max-pool and `icemapper.ICEmapper_v2(...)` for time-weighted pool
- `timesteps`, `patch_size`, `batch_size`, `learning_rate`, etc.

Once set up, run:
```
python train.py
```

Outputs:
- best weights -> `weights/<model_name>.h5`
- training logs -> `logs/<model_name>.csv`


### Evaluation on HDF5 splits

`predict_and_evaluate.py` runs inference tile-by-tile on an HDF5 and writes:
- per-tile softmax scores into an output HDF5
- a pickled evaluation summary

The usage is as follows:
```
usage: predict_and_evaluate.py [-h] [-n MODEL_NAME] [-f [FEATURES ...]] [-v {v1,v2}]

options:
  -h, --help            show this help message and exit
  -n MODEL_NAME, --model_name MODEL_NAME
                        Model name
  -f [FEATURES ...], --features [FEATURES ...]
                        Input features
  -v {v1,v2}, --version {v1,v2}
                        ICEmapper version
```
Example:
```
python predict_and_evaluate.py \
  -n 20241009_ICEmapper_v2_grdinsar \
  -f grd insar \
  -v v2
```

Important:
- Adjust `dataset_path = ...` accordingly
- Make sure the output folder exists (`output_probs_path = ...` and `output_eval_path = ...`)


### Confidence calibration and uncertainty

The repository also includes optional components used in the manuscript's uncertainty analysis:
- `confidence_calibration.ipynb`&mdash;calibrates pixel-wise confidence
- `block_bootstrapping.py`&mdash;performs block bootstrapping to estimate uncertainty of aggregated glacier area (calibration included)
- `rates_resampling/reevaluation*_*.py`&mdash;resamples historical area change rates


### Applying ICEmapper to your own GeoTIFF time series

Use `apply.py` to run a pretrained model on a one-year time series of GRD + InSAR coherence rasters.
Requirements:
- exactly 15 GRD rasters and 15 InSAR coherence rasters
- same spatial domain; the script will reproject inputs to a reference raster

The usage is as follows:
```
usage: apply.py [-h] [--grd_paths [GRD_PATHS ...]] [--insar_paths [INSAR_PATHS ...]] [--ref_path REF_PATH] [--model_name MODEL_NAME] output_path

positional arguments:
  output_path           Folder to save .tif probs

options:
  -h, --help            show this help message and exit
  --grd_paths [GRD_PATHS ...]
                        GRD scene paths
  --insar_paths [INSAR_PATHS ...]
                        InSAR scene paths
  --ref_path REF_PATH   Reference raster path for reprojection
  --model_name MODEL_NAME
                        Model name
```


### Pretrained models

Luckily, the number of parameters is small enough, so the weights are directly deposited in this repository. 
You will find them in [the weights folder](weights). 
The file names follow the template `ICEmapper_<maxpool|tweightedpool>_<FEATURES,>.h5`. 


### Ice divides

We moved the algorithms for ice divides reconstruction to [a different repository](https://github.com/konstantin-a-maslov/massive_ice_divides).


## License

This software is licensed under the [GNU General Public License v2](LICENSE).



## Citing

**!To be updated once the journal paper is published!**

To cite the paper/repository, please use the following bib entry. 

<!-- TODO: Update BibTeX once published in a journal```
@article{INDEX,
    title={TITLE},
    author={Maslov, Konstantin A. and Schellenberger, Thomas and Persello, Claudio and Stein, Alfred},
    journal={},
    year={YEAR},
    volume={},
    number={},
    pages={},
    doi={}
}
``` -->
```
@article{icemapper2025,
    title={Three times accelerated glacier area loss in Svalbard revealed by deep learning},
    author={Maslov, Konstantin A. and Schellenberger, Thomas and Persello, Claudio and Stein, Alfred},
    year={2025},
    archivePrefix={EarthArXiv},
    doi={10.31223/X5472T},
    url={https://doi.org/10.31223/X5472T},
}
```
