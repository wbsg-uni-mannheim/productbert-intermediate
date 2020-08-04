# productbert-intermediate

This repository contains code and data download scripts for the paper [Intermediate Training of BERT for Product Matching](http://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/v2/papers/DI2KG2020_Peeters.pdf) by Ralph Peeters, Christian Bizer and Goran Glavaš

* **Requirements**

    [Anaconda3](https://www.anaconda.com/products/individual)

    Please keep in mind that the code is not optimized for portable or even non-workstation devices. Some of the scripts require large amounts of RAM (64GB+). It     it advised to use a powerful workstation or server when experimenting with some of the large files.

    The code has only been used and tested on Linux (Manjaro, Ubuntu, CentOS) computers.

* **Building the conda environment**

    To build the exact conda environment used for the experiments, navigate to the project root folder where the file *productbert-intermediate.yml* is located and run ```conda env create -f productbert-intermediate.yml```
    
    Furthermore you need to install the project as a package. To do this, activate the productbert-intermediate environment with ```conda activate productbert-intermediate```, navigate to the root folder of the project, and run ```pip install -e .```

* **Downloading the raw data files and intermediate trained models**

    Navigate to the *src/data/* folder and run ```python download_datasets.py``` to automatically download the files into the correct locations.
    You can find the data at *data/raw/* and the intermediate trained models at *src/productbert/saved/models*

* **Processing the data**

    To prepare the data for the experiments, run the following scripts in that order. Make sure to navigate to the respective folders first.
    
    1. *src/processing/preprocess/preprocess_corpus.py*
    2. *src/processing/preprocess/preprocess_ts_gs.py*
    3. *src/processing/process-bert/process_to_bert.py*
    4. *src/processing/process-magellan/process_to_magellan.py*
    5. *src/processing/process-wordcooc/process-to-wordcooc.py*

    Note that some of the scripts contain global variables near the top that you can set to control what data should be processed. Depending on your needs you may not want to preprocess everything as this can take a long time, especially when processing the extremely large intermediate training sets.
    
    Furthermore the folder *src/processing/sample-training-sets* contains script to replicate building the intermediate training sets. These are provided in the download file, so this step is unnecessary if you are only interested in using them.
    If you want to replicate the building process run the files in this order and change the global variables to fit the training set you want to build:

    1. *sample_intermediate_training_sets.py*
    2. *process_intermediate_training_sets.py*
    3. *build_intermediate_training_sets.py*
    
* **Running the baseline experiments**

    Run the following scripts to replicate the baseline experiments:
    * **Magellan**:
        Navigate to *src/models/magellan/* and run the script *run_magellan.py*
    * **Word Coocurrence**:
    Navigate to *src/models/wordcooc/* and run the script *run_wordcooc.py*
    * **Deepmatcher**:
    Navigate to *src/models/deepmatcher* and run any of the scripts *train_computers_\*.py* to run the best hyperparameter setting 3 times.
    You can also adjust the used hyperparameters in said files.
    
        To allow for gradient updates (fine-tuning) of the embedding layer, simply change the line ```embed.weight.requires_grad = False``` in *models/core.py* to ```True``` in the deepmatcher package
    
    Result files can then be found in the *reports* folder.

* **Running the BERT experiments**

    Navigate to *src/productbert/*
    This project is based on a <a target="_blank" href="https://github.com/victoresque/pytorch-template/">PyTorch template project</a> It is suggested to read the respective github readme to understand how to train models and possible input commands.
    * **Fine-Tuning**:
    The folder *src/productbert* contains bash scripts to run all of the experiments including the learning rate sweeps. Run any of the bash scripts titled *train_computers_\*.sh* and append the id of the gpu you want to use, e.g. ```bash train_computers_small.sh 0```
    * **Intermediate Training**:
    The intermediately trained models are offered as part of the data download if you simply want to further use them. For replicating the intermediate training, run ```python train.py --device 0,1,2,3 -c config_category_pretrain.json``` while replacing category with either *computers*, *4cat* or *computers_mlm*. You may have to adjust the n_gpus parameter and the --device argument depending on the numbers of gpus available to you.
    
        After finishing the 36th epoch, resume training from that epoch using the --resume argument using the config config_category_pretrain_cont.json, e.g. ```python train.py --device 0,1,2,3 -c config_category_pretrain_cont.json --resume path/to/checkpoint``` This is due to the first 36 epochs being trained on sequence lengths of 128 and the last 4 on lengths of 512.
    * **Evaluating a trained model on a test set**:
    This is done by providing a config containing configuration parameters for the test run. Additionally you need to provide the checkpoint that should be used for testing. An example would be ```python test.py --device 0 -c config_computers_small_test.json --resume saved/models/pathto/model/model_best.pth```
    
    The results of the BERT experiments can then be found in *src/productbert/saved/log* and the respective model checkpoints in *src/productbert/saved/models*.
    
    **NOTE**: When adjusting the batch size in any of the configs, make sure to also adjust the number of accumulation steps, as the combination of both constitutes the actual batch size.


--------

Project based on the [cookiecutter data science project template](https://drivendata.github.io/cookiecutter-data-science/). #cookiecutterdatascience

PyTorch Project based on the [PyTorch template project](https://github.com/victoresque/pytorch-template/) by [Victor Huang](https://github.com/victoresque).
