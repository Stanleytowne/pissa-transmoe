BASE_MODEL="/data/tpz/huggingface/meta-llama/Llama-2-7b-hf"
OUTPUT_PATH='output/test'
DATA_PATH="pissa-dataset"
export HF_ENDPOINT=https://hf-mirror.com

torchrun --nproc_per_node=4 utils/gen_transformers.py --model $BASE_MODEL --sub_task metamath --output_file $OUTPUT_PATH/metamath_response.jsonl --batch_size 32 --num_gpus 4
python utils/test_acc.py --input_file $OUTPUT_PATH/metamath_response.jsonl
