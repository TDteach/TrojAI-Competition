#!/bin/bash

ROOT=$HOME/share/trojai
ROUND=$ROOT/round12
PHRASE=$ROUND/cyber-pdf-dec2022-train
MODELDIR=$PHRASE/models



# python3 example_trojan_detector.py --configure_mode --configure_models_dirpath $MODELDIR

# python3 example_trojan_detector.py --model_filepath $MODELDIR/id-00000001/model.pt --round_training_dataset_dirpath $MODELDIR

python entrypoint.py configure \
    --scratch_dirpath ./scratch/ \
    --metaparameters_filepath ./metaparameters.json \
    --schema_filepath ./metaparameters_schema.json \
    --learned_parameters_dirpath ./learned_parameters/ \
    --configure_models_dirpath $MODELDIR \
    --scale_parameters_filepath ./scale_params.npy \
#    --automatic_configuration

exit

#echo "rm learned_parameters"
#rm -rf learned_parameters
#echo "mv new_learned_parameters to learned_parameters"
#mv new_learned_parameters learned_parameters

python entrypoint.py infer \
    --model_filepath $MODELDIR/id-00000000/model.pt \
    --result_filepath ./scratch/output.txt \
    --scratch_dirpath ./scratch \
    --examples_dirpath $MODELDIR/id-00000000/clean-example-data \
    --round_training_dataset_dirpath $PHRASE \
    --learned_parameters_dirpath ./learned_parameters \
    --metaparameters_filepath ./metaparameters.json \
    --schema_filepath ./metaparameters_schema.json \
    --scale_parameters_filepath ./scale_params.npy



