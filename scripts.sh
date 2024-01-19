#!/bin/bash

ROOT=$HOME/share/trojai
ROUND=$ROOT/round16
PHRASE=$ROUND/rl-randomized-lavaworld-aug2023-train
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
    --configure_models_dirpath $MODELDIR \
    --tokenizers_dirpath tokenizers
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
    --configure_models_dirpath $MODELDIR \
    --tokenizers_dirpath haha
fi


#echo "rm learned_parameters"
#rm -rf learned_parameters
#echo "mv new_learned_parameters to learned_parameters"
#mv new_learned_parameters learned_parameters


# id-046 DetrForObjectDetection for misclassification
# id-011 FasterRCNN for misclassification
# id-049 FasterRCNN for evasion
# id-051 SSD for localization: shifting the bbox to somewhere else than the correct location
# id-061 FasterRCNN for injection: adding a new kind of detection objects


if [ $(( $1 & 2 )) -gt 0 ]
then
echo inference id-$(printf "%08d" $2)
CUDA_VISIBLE_DEVICES=3 python entrypoint.py infer \
    --model_filepath $MODELDIR/id-$(printf "%08d" $2)/model.pt \
    --result_filepath ./scratch/output.txt \
    --scratch_dirpath ./scratch \
    --examples_dirpath $MODELDIR/id-$(printf "%08d" $2)/poisoned-example-data \
    --round_training_dataset_dirpath $PHRASE \
    --learned_parameters_dirpath ./learned_parameters \
    --metaparameters_filepath ./metaparameters.json \
    --schema_filepath ./metaparameters_schema.json \
    --tokenizer_filepath tokenizers
fi



