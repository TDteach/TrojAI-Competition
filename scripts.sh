#!/bin/bash

ROOT=$HOME/share/trojai
ROUND=$ROOT/round11
PHRASE=$ROUND/image-classification-sep2022-train
MODELDIR=$PHRASE/models



# python3 example_trojan_detector.py --configure_mode --configure_models_dirpath $MODELDIR

python3 example_trojan_detector.py --model_filepath $MODELDIR/id-00000001/model.pt --round_training_dataset_dirpath $MODELDIR
