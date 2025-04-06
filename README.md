![LOGO](https://github.com/DeepWave-KAUST/ElasticSiameseFWI/blob/main/asset/Fig1.png)

Reproducible material for **DW0078, Enhancing Multi-parameter Elastic Full Waveform Inversion with a Siamese Network - Omar M. Saad and Tariq Alkhalifah**

[Click here](https://kaust.sharepoint.com/:f:/r/sites/M365_Deepwave_Documents/Shared%20Documents/Restricted%20Area/REPORTS/DW0078?csf=1&web=1&e=sBwoLF) to access the Project Report. Authentication to the _Restricted Area_ filespace is required.

# Project structure
This repository is organized as follows:

* :open_file_folder: **asset**: folder containing logo;
* :open_file_folder: **data**: folder containing data. Please note that the Volve field data is uploaded to the restricted area;
* :open_file_folder: **Model**: folder containing the Siamese network;
* :open_file_folder: **utils**: folder containing the utilities;

## Notebooks
The following notebooks are provided:

- :orange_book: ``ElasticSiameseFWI_BPSalt.ipynb``: notebook for ElasticSiameseFWI framework for BP Salt model;
- :orange_book: ``ElasticSiameseFWI_MultiSource_Overthrust.ipynb``: notebook for ElasticSiameseFWI multi-source framework for Overthrust model;
- :orange_book: ``ElasticSiameseFWI_SeamArid_NormalizedLoss.ipynb``: notebook for ElasticSiameseFWI framework for Seam Arid model using normalized loss function;
- :orange_book: ``ElasticSiameseFWI_Volve_FieldData_4Hz.ipynb``: notebook for ElasticSiameseFWI framework for Volve field data using normalized loss function;



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
DW0078 - Omar M. Saad and Tariq Alkhalifah (2025) Enhancing Multi-parameter Elastic Full Waveform Inversion
with a Siamese Network


