
GPU=1
OUTPUT_DIR=""

DATASET="fgvc_aircraft"

NIPC=500
MIN_LR=1e-8
EPOCH=50
WARMUP_EPOCH=0
IS_MIX_AUG=False
BS=128

FEWSHOT_SEED="seed_0"
N_SHOT=16
N_TEMPLATE=1

IS_SYNTH_TRAIN=True
IS_DATASET_WISE=True
DD_LR=1e-4
DD_EP=200
DD_TTE=True

IS_POOLED=True
LAMBDA_1=0.8

CE_REAL=4.0
CE_SYN=1.0
LAM_DIS=20.0
LAM_ROB=200.0
NUM_CENTROIDS=200.0
NUM_ITER=300
LEARNING_RATES=("1e-4")
WEIGHT_DECAYS=("5e-4")
MIX_AUG_SETTINGS=("True")
for LR in "${LEARNING_RATES[@]}"; do
    for WD in "${WEIGHT_DECAYS[@]}"; do
        for IS_MIX_AUG in "${MIX_AUG_SETTINGS[@]}"; do
            CUDA_VISIBLE_DEVICES=$GPU python main.py \
            --model_type=clip \
            --output_dir=$OUTPUT_DIR \
            --n_img_per_cls=$NIPC \
            --is_lora_image=True \
            --is_lora_text=True \
            --is_synth_train=$IS_SYNTH_TRAIN \
            --sd_version="sd2.1" \
            --n_template=$N_TEMPLATE \
            --guidance_scale=2.0 \
            --is_pooled_fewshot=$IS_POOLED \
            --lambda_1=$LAMBDA_1 \
            --epochs=$EPOCH \
            --warmup_epochs=$WARMUP_EPOCH \
            --log=wandb \
            --wandb_project=datadream \
            --dataset=$DATASET \
            --n_shot=$N_SHOT \
            --lr=$LR \
            --wd=$WD \
            --min_lr=$MIN_LR \
            --fewshot_seed=$FEWSHOT_SEED \
            --is_mix_aug=$IS_MIX_AUG \
            --is_dataset_wise=$IS_DATASET_WISE \
            --datadream_lr=$DD_LR \
            --datadream_epoch=$DD_EP \
            --datadream_train_text_encoder=$DD_TTE \
            --batch_size=$BS \
            --ce_real=$CE_REAL \
            --ce_syn=$CE_SYN \
            --lam_dis=$LAM_DIS \
            --lam_rob=$LAM_ROB \
            --num_centroids=$NUM_CENTROIDS \
            --num_iters=$NUM_ITER \
            $PARAM
        done
    done
done