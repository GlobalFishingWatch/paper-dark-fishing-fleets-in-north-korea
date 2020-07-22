# Pair Vessel Optical Vessel Detection

## Overview

Note that we cannot supply imagery for training or testing. However, 
we can supply weights for running inference on request. 

### Directory Structure

- optical_vessel_detection : the pair trawler detection models.
    * core : somewhat generic building blocks code for the models. 
    * support : support code for training and evaluating models
- notebooks : Jupyter notebooks for model setup and evaluation
- data : directory where various types of data are downloaded.


### Workflow

1. Setup environment. Run `pip install -e .`. You will
   also need to install a version of Tensorflow < 2, but you will
   need to decide whether to install the gpu or non-gpu version
   based on your environment.

2. Generate tiles. See `notebooks/SetUpPairTrawlerTraining.ipynb`. Unfortunately,
   we cannot supply the training tiles due to licensing restrictions, so
   you would need to annotate your own tiles.

3. Run training (see below). Note that there are some assumptions
   about where the tiles live built into `opticial_vessel_detection/core/tile_data.py`,
   so that file will need to be edited to fit your local setup.

4. Run inference (see below).

5. Run evaluation. See the notebooks 


## Training

We train using Google's ML Engine. The details of launching a training run
are taken care of by `deploy_cloudml.py`.  For example as typical training
run would be started using:

    python  deploy_cloudml.py \
                --model_name pair_trawler_model_v1_1 \
                --job_name run_annotatins \
                --config_file deploy_4p100.yaml

The actual command used to train the model used in the paper was:

    python  deploy_cloudml.py \
                     --model_name pair_trawler_model_v1_1
                     --job_name v1_1_aug 
                     --config_file deploy_4p100.yaml \
                     --split -1 

Where `--split -1` causes the training routine to use all of the training patches
during training, so there is no runtime validation. For more details run 
`python deploy_cloudml.py --help`


## Inference

We run inference on Google Compute Engine GPU instances.  Setting of up an instance is 
somewhat involved – see below -- however, once an instance is set up, inference is 
mostly automated. A typical inference run is started as shown:

     python -m source.process_nk_images \
                     --source gs://PATH/TO/IMAGE/SOURCE/ \
                     --target gs://PATH/TO/IMAGE/DESTINATION \
                     --detections-table BIG_QUERY_DATASET.TABLE \
                     --date 2018-09-29

Note that the data is read from `SOURCE/raw` and written to `TARGET/detections`, where `SOURCE` and 
`TARGET` are the locations specified by `--source` and `--target` respectively. So, the 
specified source and target locations can be identical without clashes. `--detections-table` specifies
a Google BigQuery table to write the results to.


### Setting UP a GPU Instance

This is specific to setting up an instance on Google Compute Engine. If setting
up on another service, you'll need to customize these directions appropriately.

1. Create an instance on GCE:
    - GPU availability is shown here: https://cloud.google.com/compute/docs/gpus/
      Notably they are available in us-west1-a/b and us-central-1a/b/f and europe-west-4a/b/c
    - Ubuntu 16.04
    - 8-core HiMem, 1 V-100
    - 2 TB drive (maybe less once we move to reading writing to GCS)
    - allow full access to all google APIs
    - After creating instance, add `ssh-server` to `Network tags` 

2. Add swap:
    - Generally follow https://tecadmin.net/enable-swap-on-ubuntu/

        sudo apt-get install -y vim
        sudo apt-get install -y git
        sudo apt-get install -y tmux
        sudo apt-get install -y bzip2

        sudo fallocate -l 128G /swapfile
        sudo chmod 600 /swapfile
        sudo mkswap /swapfile
        sudo swapon /swapfile
        sudo swapon -s

        sudo vim /etc/fstab  # add `/swapfile   none    swap    sw    0   0`
        sudo vim /etc/sysctl.conf # Add vm.swappiness=10

        sudo sysctl -p

3. Install Software and driver:

- copy setup script from https://cloud.google.com/compute/docs/gpus/add-gpus#install-gpu-driver, cat into `setup_gpu.sh`
      and execute using `sudo bash setup_gpu.sh`
- Then:

        wget https://repo.anaconda.com/miniconda/Miniconda2-latest-Linux-x86_64.sh
        bash Miniconda2-latest-Linux-x86_64.sh -b -p $HOME/miniconda
        printf "\nexport PATH=\"\$HOME/miniconda/bin:\$PATH\"\n" >> ~/.bashrc
        source ~/.bashrc
        conda create --name dff2 python=3.7 "tensorflow-gpu<2" gdal
        conda activate dff2
        sudo apt-get install fonts-freefont-ttf 

4. Install  `paper-dark-fishing-fleets-in-north-korea`:

        git clone https://github.com/GlobalFishingWatch/paper-dark-fishing-fleets-in-north-korea.git
        cd paper-dark-fishing-fleets-in-north-korea/supplementary-sections/2-daytime-optical-imagery
        pip install -e .

5. Test:

        python -m optical_vessel_detection.support.process_nk_images \
            --source gs://PATH/TO/IMAGE/SOURCE/ \
            --target gs://PATH/TO/IMAGE/DESTINATION \
            --detections-table BIG_QUERY_DATASET.TABLE \
            --date DATE_TO_PROCESS
