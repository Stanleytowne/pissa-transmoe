BASE_MODEL="/ceph/home/muhan01/tpz/transmoe/alpaca/output/alpacamoe-2-7B"
OUTPUT_PATH="output/metamath-OnlyRouter-LlamaMoe-2-7b"
DATA_PATH="/ceph/home/muhan01/tpz/PiSSA/pissa-dataset"
export HF_ENDPOINT=https://hf-mirror.com

# batch size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus = 128
deepspeed --master_port=16971 --include=localhost:0,1,2,3 train.py \
    --deepspeed configs/ds_config_zero2_no_offload.json \
    --model_name_or_path $BASE_MODEL \
    --full_finetune True \
    --router_only True \
    --bf16 \
    --data_path $DATA_PATH \
    --sub_task metamath \
    --dataset_split "train"\
    --dataset_field instruction output \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 1 \
    --model_max_length 512 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard"


python utils/gen_transformers.py --model $OUTPUT_PATH --sub_task metamath --output_file $OUTPUT_PATH/metamath_response.jsonl --batch_size 8
python utils/test_acc.py --input_file $OUTPUT_PATH/metamath_response.jsonl
