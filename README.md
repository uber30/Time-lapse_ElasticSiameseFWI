![LOGO](https://github.com/DeepWave-KAUST/ElasticSiameseFWI/blob/main/asset/Fig1.png)

Official reproducible material for **Enhancing Multi-parameter Elastic Full Waveform Inversion with a Siamese Network - Omar M. Saad and Tariq Alkhalifah**



# Project structure
This repository is organized as follows:

* :open_file_folder: **asset**: folder containing logo;
* :open_file_folder: **data**: folder containing data. Please note that the Volve field data is uploaded to the restricted area;
* :open_file_folder: **Model**: folder containing the Siamese network;
* :open_file_folder: **utils**: folder containing the utilities;

## Notebooks
The following notebooks are provided:


- :orange_book: ``ElasticSiameseFWI_SeamArid_NormalizedLoss.ipynb``: notebook for ElasticSiameseFWI framework for Seam Arid model using normalized loss function;



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

## Cite us 
```bibtex
@article{saad2025siamesefwi,
  title={Enhancing Multi-parameter Elastic Full Waveform Inversion with a Siamese Network},
  author={Saad, Omar M and Alkhalifah, Tariq},
  journal={The Leading Edge},
  volume={1},  % Replace with actual volume number
  number={1},  % Replace with actual issue number
  pages={1-10}, % Replace with actual page range
  year={2025},
  doi={https://doi.org/xxxxx}, % Replace with actual DOI
  publisher={Society of Exploration Geophysicists}
}

