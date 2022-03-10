Code and data used in the "GAN-Aimbots: The Next Generation of Cheating in Multiplayer Video Games"

**Note:** Code quality is best classified as "researchy", mixed with bad case of not following code styles. 

For data, see [this Zenodo](https://zenodo.org/record/6345323) record.

## Contents and overview

* `data` directory contains the data collected during the research (download from [Zenodo](https://zenodo.org/record/6345323)).
* `scripts` contains shell scripts that should be used to run experiments.
* `shared_parameters` contains the group 1 and 2 GAN-Aimbots used in the paper. They are also used by the recording scripts (see below).
* `scenarios` contains the Doom map/scenario files for setting up the game.
* Remaining directories/files are Python files that contain all the code for the experiments.

## Requirements

Playing with the aimbots in ViZDoom *only* works on Windows!

Experiments were ran on Python 3.6, but should work on any Python 3.x. For other requirements, see `requirements.txt`.

**Note**: To compare different classifiers, you also need to install [auto-sklearn](https://automl.github.io/auto-sklearn/master/).

## Running the experiments

To repeat the experiments in paper with shared data, run `./scripts/run_all.sh` in the root directory.
If all works out, this should train DNN classifiers in different setups and evaluate them, and results should
match closely to what was reported in the paper. This will also train the GAN-Aimbots, but they are not used
for the results, as that would require data collection.

The whole process should take well less than a hour. No GPU is needed.

All results will be printed out in the console window and figures are placed under `figures` directory.

## Trying out the game

Run one of the `data_collection_*` scripts to try out the game and different aimbots. For a newcomer,
we recommend playing `data_collection_performance.py` (takes 30min) which will let you play with and without
all aimbots used in the work.

## Data collection scripts

Scripts starting with `data_collection_*` are Python scripts that were used to gather the data,
in their original form (including the messages to users). These were packed into single executable
files (.exe) using PyInstaller.

Launching any of the following scripts prompts user to agree with a consent agreement, and after
agreeing launches a Doom game with or without aimbots. Data is stored in the same directory
where script was launched.

* `data_collection_original.py` is data collection 1, collecting data for light and strong aimbots (and no aimbot).
* `data_collection_gan.py` is data collection 2, collecting data for GAN group 1 and 2 aimbots (and no aimbot)
* `data_collection_performance.py` is data collection 3, collecting data for light, strong and GAN group 1 aimbots (and no aimbot).
* `data_collection_recording.py` is data collection 4, collecting video recordings of light, strong and GAN group 1 aimbots (and no aimbot).




