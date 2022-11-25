![Title](/images/weather4cast_v1000-26.png?raw=true "Weather4cast competition")

# [Weather4cast](https://www.iarai.ac.at/weather4cast/)  - Super-Resolution Rain Movie Prediction under Spatio-Temporal Shifts
- Predict super-resolution rain movies  in various regions of Europe
- Transfer learning across space and time under strong shifts
- Exploit data fusion to model ground-radar and multi-band satellite images

## Contents
[Weather4cast: Super-Resolution Rain Movie Prediction under Spatio-Temporal Shifts](#weather4cast-multi-sensor-weather-forecasting-competition--benchmark-dataset)
  - [Contents](#contents)
  - [Introduction](#introduction)
  - [Get the data](#get-the-data)
  - [Submission guide](#submission-guide)
  - [Starter kit](#starter-kit)
    - [Environment](#environment)
    - [Training](#training)
    - [Validation](#validation)
    - [TensorBoard](#tensorboard)
    - [Generating a submission](#generating-a-submission)
    - [Automated generation of submissions](#automated-generation-of-submissions-helper-scripts)
  - [Code and Scientific Abstract](#code-and-scientific-abstract)
  - [Cite](#citation)
  - [Credits](#credits)
  
## Introduction
The aim of the 2022 edition of the Weather4cast competition is to predict future high resolution rainfall events from lower resolution satellite radiances. Ground-radar reflectivity measurements are used to calculate pan-European composite rainfall rates by the [Operational Program for Exchange of Weather Radar Information (OPERA)](https://www.eumetnet.eu/activities/observations-programme/current-activities/opera/) radar network. While these are more precise, accurate, and of higher resolution than satellite data, they are expensive to obtain and not available in many parts of the world. We thus want to learn how to predict this high value rain rates from radiation measured by geostationary satellites operated by the [European Organisation for the Exploitation of Meteorological Satellites (EUMETSAT)](https://www.eumetsat.int/). 

# Prediction task
Competition participants should predict rainfall locations for the next 8 hours in 32 time slots from an input sequence of 4 time slots of the preceeding hour. The input sequence consists of four 11-band spectral satellite images. These 11 channels show slightly noisy satellite radiances covering so-called visible (VIS), water vapor (WV), and infrared (IR) bands. Each satellite image covers a 15 minute period and its pixels correspond to a spatial area of about 12km x 12km. The prediction output is a sequence of 32 images representing rain rates from ground-radar reflectivities. Output images also have a temporal resolution of 15 minutes but have higher spatial resolution, with each pixel corresponding to a spatial area of about 2km x 2km. So in addition to predicting the weather in the future, converting satellite inputs to ground-radar outputs, this adds a super-resolution task due to the coarser spatial resolution of the satellite data

### Weather4cast 2022 - Stage 1
For Stage 1 of the competition we provide data from three Eureopean regions selected based on their preciptation characteristics. The task is to predict rain events 8 hours into the future from a 1 hour sequence of satellite images. The models should output binary pixels, with 1 and 0 indicating *rain* or *no rain* respectively. Rain rates computed from OPERA ground-radar reflectivities provide a ground truth. Although we provide the rain rates, at this stage, only rain/no rain needs to be predicted for each pixel.

For Stage 1 we provide data from one year only, covering February to December 2019. 

### Weather4cast 2022 - Stage 2
In Stage 2 additional data will be provided for 2020 and 2021. Years 2019 and 2020 can then be used for training, while test sets from 2021 assess model robustness to temporal shifts. Additional regions with different climatological characteristics test model robustsness under spatial shifts. There are thus additional files for the new regions and years and thus the folder structure for stage 2 has been expanded accordingly to include additional sub-folders with the data for 2020 and 2021. In total there then are 7 regions with full training data in both 2019 and 2020. Three additional regions provide a spatial transfer learning challenge in years 2019 and 2020. For all ten regions, the year 2021 provides a temporal transfer learning challenge. For the seven regions with extensive training data in 2019 and 2020 this constitutes a pure temporal transfer learning challenge. The three additional regions 2021 provide a combined spatial and temporal transfer learning challenge.


## Get the data
You need to register for the competition and accept its Terms and Conditions to access the data:

- Competition Data: [Join and get the data](https://www.iarai.ac.at/weather4cast/get-data-2022/)

Data are provided in [HDF-5](https://docs.h5py.org/en/stable/quick.html) files, separately for each year and data type. In our canonical folder structure `year/datatype/` the HRIT folder holds the satellite data and the OPERA folder provides the ground radar data. The file names reflect the different regions (`boxi_####`) and data splits (`train`, `validation`, and `test`). Ground truth for the test data split is of course withheld.

After downloading the data, your data files should thus be arranged in folders of the following structure:
```
2019/
    +-- HRIT/  ... sub-folder for satellite radiance datasets
        +-- boxi_0015.test.reflbt0.ns.h5
        +-- boxi_0015.train.reflbt0.ns.h5
        +-- boxi_0015.val.reflbt0.ns.h5
        +-- boxi_00XX…….
    +-- OPERA/  ... sub-folder for OPERA ground-radar rain rates
        +-- boxi_0015.train.rates.crop.h5
        +-- boxi_0015.val.rates.crop.h5
        +-- boxi_00XX…….
2020/
    +-- HRIT/  ... sub-folder for satellite radiance datasets
        +-- boxi_0015.test.reflbt0.ns.h5
        +-- boxi_0015.train.reflbt0.ns.h5
        +-- boxi_0015.val.reflbt0.ns.h5
        +-- boxi_00XX…….
    +-- OPERA/  ... sub-folder for OPERA ground-radar rain rates
        +-- boxi_0015.train.rates.crop.h5
        +-- boxi_0015.val.rates.crop.h5
        +-- boxi_00XX…….  
```

Each HDF file provides a set of (multi-channel) images: 

- **boxi_00XX.train.reflbt0.ns.h5** provides *REFL-BT*, which is a tensor of shape `(20308, 11, 252, 252)` representing 20,308 images with 11 channels of satellite radiances for region XX. These are the input training data. The order of the channels in the H5 file corresonds to the following order of the satellite channels: `IR_016, IR_039, IR_087, IR_097, IR_108, IR_120,IR_134, VIS006, VIS008, WV_062, WV_073`. 

- **boxi_00XX.train.rates.crop.h5** provides *rates.crop*, which is a tensor of shape `(20308, 11, 252, 252)` representing OPERA ground-radar rain rates for the corresponding satellite radiances from the train dataset. Model output should be 1 or 0 for rain or no-rain predictions respectively.

- **boxi_00XX.val.reflbt0.ns.h5** provides *REFL-BT*, which is a tensor of shape `(2160, 11, 252, 252)` representing additional measured satellite radiances. This data can be used as input for independent model validation. There are 60 validation sequences and each validation sequence consists of images for 4 input time slots; while in addition we also provide images for the 32 output time slots please note that this is just to aid model development and that model predictions cannot use these. The validation data set thus holds 4x60 + 32x60 = 2,160 images in total.

- **boxi_00XX.val.rates.crop.h5** provides *rates.crop*, which is a tensor of shape `(2160, 1, 252, 252)` representing OPERA ground-radar rain rates for the corresponding satellite radiances from the validation dataset. Model output should be 1 or 0 for rain or no-rain predictions respectively.

- **boxi_00XX.test.reflbt0.ns.h5** provides *REFL-BT*, which is a tensor of a shape `(240, 11, 252, 252)` representing additional satellite radiances. This dataset gives the input data for your model predictions for submission to the leaderboard. There are 60 input sequences in total, as each test sequence consists of images for 4 time slots (4x60 = 240). Note that no satellite radiances are provided for the future, so this is a true prediction task.

Both input satellite radiances and output OPERA ground-radar rain rates are given for 252x252 pixel patches but please note that the spatial resolution of the satellite images is about six times lower than the resolution of the ground radar. This means that the 252x252 pixel ground radar patch corresponds to a 42x42 pixel center region in the coarser satellite resolution. The model target region thus is surrounded by a large area providing sufficient context as input for a prediction of future weather. In fact, fast storm clouds from one border of the input data would reach the center target region in about 7-8h.

![Context](/images/opera_satelite_context_explained.png?raw=true "Weather4cast competition")

- Stage-2 Competition: [Join and get the data](https://www.iarai.ac.at/weather4cast/get-data-2022/)
## Submission guide
For submissions you need to upload a ZIP format archive of HDF-5 files that follows the folder structure below. Optionally, each HDF-5 file can be compressed by gzip, allowing for simple parallelization of the compression step. You need to include model predictions for all the regions. For each region, an HDF file should provide *submission*, a tensor of type `float32` and shape `(60, 1, 32, 252, 252)`, representing your predictions for the 60 test samples of a region. You need to follow the file naming convention shown in the example below to indicate the target region. Predictions for different years need to be placed in separate folders as shown below. The folder structure must be preserved in the submitted ZIP file. Please note that for Stage 1 we only ask for predictions for the year 2019, and predictions are simply 1 or 0 to indicate *rain* or *no rain* events respectively. For the Stage 2 Core Challenge, we ask for predictions for a total of 7 regions in both 2019 and 2020. For the Stage 2 Transfer Learning Challenge, predictions for 3 regions are required in years 2019 and 2020, and for all 10 regions in 2021. To simplify compilation of predictions, we now provide helper scripts in the Starter Kit.

```
+-- 2019 –
    +-- boxi_0015.pred.h5.gz   ...1 file per region for 60 test-sequence predictions of 32 time-slots each
    +-- boxi_00XX….
+-- 2020 –
    +-- boxi_0015.pred.h5.gz  
    +-- boxi_00XX….
```

## Starter kit
This repository provides a starter kit accompanying the Weather4cast 2022 competition that includes example code to get you up to speed quickly. Please note that its use is entirely optional. The sample code includes a dataloader, some helper scripts, and a Unet-3D baseline model, some parameters of which can be set in a configuration file.

To obtain the baseline model, you will need the `wget` command installed - then you can run
```
./mk_baseline.sh
```
to fetch and patch a basic 3D U-Net baseline model.

You will need to download the competition data separately. The sample code assumes that the downloaded data are organized in the following folder structure (shown here for Stage-1 data, conversely for Stage-2):

```
+-- data
    +-- 2019 –
        +-- HRIT --
            +-- boxi_0015.test.reflbt0.ns.h5
            +-- boxi_0015.train.reflbt0.ns.h5
            +-- boxi_0015.val.reflbt0.ns.h5
            +-- boxi_0034.test.reflbt0.ns.h5
            +-- boxi_0034.train.reflbt0.ns.h5
            +-- boxi_0034.val.reflbt0.ns.h5
            +-- boxi_0076.test.reflbt0.ns.h5
            +-- boxi_0076.train.reflbt0.ns.h5
            +-- boxi_0076.val.reflbt0.ns.h5
        +-- OPERA -- 
            +-- boxi_0015.train.rates.crop.h5
            +-- boxi_0015.val.rates.crop.h5
            +-- boxi_0034.train.rates.crop.h5
            +-- boxi_0034.val.rates.crop.h5
            +-- boxi_0076.train.rates.crop.h5
            +-- boxi_0076.val.rates.crop.h5
```

The path to the parent folder `data` needs to be provided as the `data_root` parameter in the `config_baseline.yaml` file.

### Environment
We provide Conda environments for the sample code which can be recreated locally. An environment with libraries current at release can be recreated from the file `w4cNew.yml`using the following command:
```
conda env create -f w4cNew.yml
```
If you want to use older libraries for compatibility reasons, we also provide an earlier environment in `w4c.yml`. Finally, if you want to create an environment in the future, we also provide a script `mk_env.sh` to get you started.
Note that all this can easily run for an hour or more, depending on your machine and setup.

To activate the environment please run
```
conda activate w4cNew
```
or
```
conda activate w4c
```
respectively.

### Training

We provide a script `train.py` with all the necessary code to train and explore a modified version of a [3D variant of the U-Net](https://github.com/ELEKTRONN/elektronn3). The script supports training from scratch or fine tuning from a provided checkpoint. The same script can also be used to evaluate model predictions on the validation data split using the flag `--mode val`, or to generate submissions from the test data using the flag `--mode predict`.
In all cases please ensure you have set the correct data path in `config_baseline.yaml` and activated the `w4c` environment.

*Example invocations:*
- Training the model on a single GPU:
```
python train.py --gpus 0 --config_path config_baseline.yaml --name name_of_your_model
```
If you have more than one GPU you can select which GPU to use, with numbering starting from zero.
- Fine tuning the model on 4 GPUs starting with a given checkpoint:
```
python train.py --gpus 0 1 2 3 --mode train --config_path config_baseline.yaml --checkpoint "lightning_logs/PATH-TO-YOUR-MODEL-LOGS/checkpoints/YOUR-CHECKPOINT-FILENAME.ckpt" --name baseline_tune
```

### Validation
Training will create logs and checkpoint files that are saved in the `lightning_logs` directory. To validate your model from a checkpoint you can for example run the following command (here for two CPUs):
```
python train.py --gpus 0 1 --mode val  --config_path config_baseline.yaml  --checkpoint "lightning_logs/PATH-TO-YOUR-MODEL-LOGS/checkpoints/YOUR-CHECKPOINT-FILENAME.ckpt" --name baseline_validate
```

### TensorBoard
You can of course also use [TensorBoard](https://www.tensorflow.org/tensorboard) to track and visualize model evaluation metrics during the training process.
The standard TensorBoard command line is:
```
tensorboard --logdir ./lightning_logs
```
This should confirm that TensorBoard has started. For the default port, you point your browser to http://localhost:6006.

### Generating a submission
Submission files can be generated from a trained model based on the model paramters saved in the checkpoint file. To generate predictions from your model checkpoint you can run the `train.py` script as below:
```
train.py --gpus 0 --mode predict --config_path config_baseline.yaml --checkpoint "lightning_logs/PATH-TO-YOUR-MODEL-LOGS/checkpoints/YOUR-CHECKPOINT-FILENAME.ckpt"
```
The code currently does not support generating a prediction for more than one region/year at a time.

The results are saved in a single HDF-5 file named `boxi_00XX.pred.h5` in the `./submssion/YEAR/` folder, where *boxi_00XX* is the name of the region defined in the *predict* section your config file. A sample configuration is shown below:
```
predict:
  region_to_predict: boxi_0015
  year_to_predict: 2019
```
To generate predictions for multiple regions this needs to be run with a separate configuration file for each region.  

After generating prediction files for all the regions, please pack them into a single ZIP file (keeping the `year/` folder structure) and submit them to the [respective Weather4cast leaderboards](https://www.iarai.ac.at/weather4cast/challenge/).

### Automated generation of submissions (helper scripts)
Considering the much increased number of individual predictions to collect for a leaderboard submission in Stage-2, we now provide helper scripts `mk_pred_core.sh` and `./mk_pred_transfer.sh` that can be used to generate and compile all individual predictions from a single model. The scripts display help text and diagnostics. Note that the use of these scripts is entirely optional because you may prefer to apply different models for different regions. You can provide both an output directory and a GPU ID to generate multiple predictions in parallel. The script will typically run for 20-40 minutes on a recent GPU system.

Example invocation for interactive use:
```
./mk_pred_core.sh config_baseline_stage2-pred.yaml 'lightning_logs/yourModelName/checkpoints/yourCheckPointName.ckpt' yourSubmissionName 0 2>&1 | tee yourSubmission.core.log
```

## Code and Scientific Abstract
At the end of the competition paricpants are required to provide:
1. A short scientific paper (Scientific Abstract) with a sufficiently detailed description of your approach (4-6 pages plus references)
2. The code and models with their learned weights that you used for your predictions, with explanations of how to reproduce you submissions.
  
We will notify participants of how to provide their scientific abstract. For the code, you will need to submit it to a public repository like GitHub, providing a link to download the model's learned weights. Ideally, your repository should at least contain:
- a) A list of **dependencies**. In the case of using Python, we suggest using conda/pip to generate them: `conda env export > environment.yml`. Make sure that your code can be executed from a fresh environment using the provided list of requirements: `conda env create -f environment.yml`.
- b) **Code**, **models**, and a **folder with all model weights**.
- c) An **out-of-the-box script** to use your best model **to generate predictions**. The script will read the inputs for the model from a given path and region, using its test folder (like the one used for the leaderboard), and save the outputs on a given path. The path to the folder containing the weights to be loaded by the models can also be an argument of the script. 


## Citation

When using or referencing the Weather4cast Competition in general or the competition data please cite: 
```
@INPROCEEDINGS{9672063,  
author={Herruzo, Pedro and Gruca, Aleksandra and Lliso, Llorenç and Calbet, Xavier and Rípodas, Pilar and Hochreiter, Sepp and Kopp, Michael and Kreil, David P.},  
booktitle={2021 IEEE International Conference on Big Data (Big Data)},   
title={High-resolution multi-channel weather forecasting – First insights on transfer learning from the Weather4cast Competitions 2021},   
year={2021},  
volume={},  
number={},  
pages={5750-5757},  
doi={10.1109/BigData52589.2021.9672063}
}

@inbook{10.1145/3459637.3482044,
author = {Gruca, Aleksandra and Herruzo, Pedro and R\'{\i}podas, Pilar and Kucik, Andrzej and Briese, Christian and Kopp, Michael K. and Hochreiter, Sepp and Ghamisi, Pedram and Kreil, David P.},
title = {CDCEO'21 - First Workshop on Complex Data Challenges in Earth Observation},
year = {2021},
isbn = {9781450384469},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3459637.3482044},
booktitle = {Proceedings of the 30th ACM International Conference on Information &amp; Knowledge Management},
pages = {4878–4879},
numpages = {2}
}
```

## Credits
The competition is organized / supported by:
- [Institute of Advanced Research in Artificial Intelligence, Austria](https://iarai.ac.at)
- [Silesian University of Technology, Poland](https://polsl.pl)
- [European Space Agency Φ-lab, Italy](https://philab.phi.esa.int/)
- [Spanish State Meteorological Agency, AEMET, Spain](http://aemet.es/)
