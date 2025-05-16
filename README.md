![LOGO](https://github.com/DeepWave-KAUST/ElasticSiameseFWI/blob/main/asset/Fig1.png)

Reproducible material for Time-Lapse multiparameter FWI, using the ElasticSiameseFWI from **Enhancing Multi-parameter Elastic Full Waveform Inversion with a Siamese Network - Omar M. Saad and Tariq Alkhalifah**


# Project structure
This repository is organized as follows:

* :open_file_folder: **asset**: folder containing logo;
* :open_file_folder: **data**: folder containing model data;
* :open_file_folder: **Model**: folder containing the Siamese **networks**;
* :open_file_folder: **utils**: folder containing the utilities (new funcs);

## Notebooks
The following notebooks are provided:


- :orange_book: ``Overthrust_parallel_cascaded_noise_constDen.ipynb``: notebook using ElasticSiameseFWI framework for the Overthrust model with the regularized parallel and cascaded approach;
- :orange_book: ``Overthrust_parallel_cascaded_noise_constDenFWI.ipynb``: notebook using EFWI for the Overthrust model with the regularized parallel and cascaded approach;
- :orange_book: ``Overthrust_joint_reg_noise.ipynb``: notebook using ElasticSiameseFWI framework for the Overthrust model with the joint inversion approach;
- :orange_book: ``Overthrust_joint_reg_noiseFWI.ipynb``: notebook using EFWI for the Overthrust model with the joint inversion approach;
- :blue_book: ``SiameseFWI_ParallelCascaded.ipynb``: notebook using SiameseFWI framework for the Overthrust model (only Vp) with the regularized parallel and cascaded approach;
- :blue_book: ``SiameseFWI_ParallelCascadedFWI.ipynb``: notebook using FWI for the Overthrust model (only Vp) with the regularized parallel and cascaded approach;
- :blue_book: ``SiameseFWI_DDWI_CDWI.ipynb``: notebook using SiameseFWI framework for the Overthrust model (only Vp) with the coupled approaches;
- :blue_book: ``SiameseFWI_DDWI_CDWI_FWI.ipynb``: notebook using FWI for the Overthrust model (only Vp) with the coupled approaches;

## Getting started :space_invader: :robot:
To ensure the reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.

Run:
```
./install_env.sh
```
It will take some time, if at the end you see the word `Done!` on your terminal you are ready to go. 

Remember to always activate the environment by typing:
```
conda activate ElasticSiameseFWI
```
**Disclaimer:** All experiments have been carried on a Intel(R) Xeon(R) CPU @ 2.10GHz equipped with a single NVIDIA GEForce RTX 3090 GPU. Different environment 
configurations may be required for different combinations of workstation and GPU.

## Cite Omar! 
```bibtex
@article{saad2025siamesefwi,
  title={Enhancing Multi-parameter Elastic Full Waveform Inversion with a Siamese Network},
  author={Saad, Omar M and Alkhalifah, Tariq},
  journal={The Leading Edge},
  volume={44},
  number={5},
  pages={416a1--416a10},
  year={2025},
  doi={doi.org/10.1190/tle44050416a1.1}, 
  publisher={Society of Exploration Geophysicists}
}

