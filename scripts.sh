#!/bin/bash

ROOT=$HOME/share/trojai
ROUND=$ROOT/round18
PHRASE=$ROUND/llm-pretrain-apr2024-train-rev2
MODELDIR=$PHRASE/models



# python3 example_trojan_detector.py --configure_mode --configure_models_dirpath $MODELDIR

# python3 example_trojan_detector.py --model_filepath $MODELDIR/id-00000001/model.pt --round_training_dataset_dirpath $MODELDIR

echo $1

if [ $1 -eq 0 ]
then
echo manual_configure
CUDA_VISIBLE_DEVICES=2 python entrypoint.py configure \
    --scratch_dirpath ./scratch/ \
    --metaparameters_filepath ./metaparameters.json \
    --schema_filepath ./metaparameters_schema.json \
    --learned_parameters_dirpath ./learned_parameters/ \
    --configure_models_dirpath $MODELDIR
fi



if [ $(( $1 & 1 )) -gt 0 ]
then
echo automatic_configure
python entrypoint.py configure \
    --automatic_configuration \
    --scratch_dirpath ./scratch/ \
    --metaparameters_filepath ./metaparameters.json \
    --schema_filepath ./metaparameters_schema.json \
    --learned_parameters_dirpath ./learned_parameters/ \
    --configure_models_dirpath $MODELDIR
fi


#echo "rm learned_parameters"
#rm -rf learned_parameters
#echo "mv new_learned_parameters to learned_parameters"
#mv new_learned_parameters learned_parameters



if [ $(( $1 & 2 )) -gt 0 ]
then
echo inference id-$(printf "%08d" $2)
CUDA_VISIBLE_DEVICES=3 python entrypoint.py infer \
    --model_filepath $MODELDIR/id-$(printf "%08d" $2) \
    --result_filepath ./output.txt \
    --scratch_dirpath ./scratch \
    --examples_dirpath $MODELDIR/id-$(printf "%08d" $2)/clean-example-data \
    --round_training_dataset_dirpath $PHRASE \
    --metaparameters_filepath ./learned_parameters/metaparameters.json \
    --schema_filepath ./metaparameters_schema.json \
    --learned_parameters_dirpath ./learned_parameters
fi



