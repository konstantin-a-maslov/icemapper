# Glacier Mapping from Sentinel-1 SAR Time Series with Deep Learning in Svalbard

[Konstantin A. Maslov](https://people.utwente.nl/k.a.maslov), [Thomas Schellenberger](https://www.mn.uio.no/geo/english/people/aca/geohyd/thosche/), [Claudio Persello](https://people.utwente.nl/c.persello), [Alfred Stein](https://people.utwente.nl/a.stein)

[[`Paper`](https://ieeexplore.ieee.org/document/10640676)] [[`Datasets`](#datasets)] [[`BibTeX`](#citing)] 

<br/>

**!The repository already contains all necessary source code and data to replicate the results presented in the manuscript. Yet, we are still working on the command line interfaces and a detailed README!**

Glaciers are one of the essential climate variables.
Tracking their areal changes over time is of high importance for monitoring the impacts of climate change and designing adaptation strategies. 
Mapping glaciers from optical remote sensing data might result in a very limited temporal resolution due to the absence of cloud-free imagery at the end of the ablation season. 
Synthetic aperture radar (SAR) solves this problem as it can operate in almost all weather conditions. 
Here, we present a deep learning strategy for glacier mapping based solely on Sentinel-1 SAR data in Svalbard. 
We test two options for integrating SAR image time series into deep learning models, namely, 3D convolutions and long short-term memory (LSTM) cells.
Both proposed models achieve an intersection over union (IoU) of 0.964 on the test subset. 
Our results highlight the applicability of SAR data in glacier mapping with the potential to obtain glacier inventories with higher temporal resolution. 


## Data access


## Installation 


## Getting started


## Pretrained models

Luckily, the number of parameters is small enough, so the weights are directly deposited in this repository. 
You will find them in [the weights folder](weights). 
The file names follow the template `ICEmapper_<maxpool|tweightedpool>_<FEATURES,>.h5`. 


## License

This software is licensed under the [GNU General Public License v2](LICENSE).


## Ice divides

We moved the algorithms for ice divides reconstruction to [a different repository](https://github.com/konstantin-a-maslov/massive_ice_divides).


## Citing

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
@inproceedings{glaciermappingfromsar2024,
    title={Glacier Mapping from Sentinel-1 SAR Time Series with Deep Learning in Svalbard}, 
    author={Maslov, Konstantin A. and Schellenberger, Thomas and Persello, Claudio and Stein, Alfred},
    booktitle={IGARSS 2024 - 2024 IEEE International Geoscience and Remote Sensing Symposium}, 
    year={2024},
    pages={14-17},
    doi={10.1109/IGARSS53475.2024.10640676}
}
```
