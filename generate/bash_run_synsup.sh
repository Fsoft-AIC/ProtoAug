GPU=2
N_SET_SPLIT=10
SPLIT_IDX=9

BS=50
NIPC=500
SD="sd2.1"
GS=2.0

N_TEMPLATE=1

MODE="datadream"
DD_LR=1e-4
DD_EP=200

DATASET="flowers102"
IS_DATASETWISE=True
FEWSHOT_SEED="seed_0"

for NS in 16; do
    echo "Running with n_shot=${NS}"
    CUDA_VISIBLE_DEVICES=$GPU python generate_checkpoint.py \
        --bs=$BS \
        --n_img_per_class=$NIPC \
        --sd_version=$SD \
        --mode=$MODE \
        --guidance_scale=$GS \
        --n_shot=${NS} \
        --n_template=$N_TEMPLATE \
        --dataset=$DATASET \
        --n_set_split=$N_SET_SPLIT \
        --split_idx=$SPLIT_IDX \
        --fewshot_seed=$FEWSHOT_SEED \
        --datadream_lr=$DD_LR \
        --datadream_epoch=$DD_EP \
        --is_dataset_wise_model=$IS_DATASETWISE
done