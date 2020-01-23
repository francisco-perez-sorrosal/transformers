#export CUDA_VISIBLE_DEVICES="0,1,2,3"
export CUDA_VISIBLE_DEVICES="2,3"

export TOXIC_DATA_DIR=./data/jigsaw-toxic-comment-classification-challenge
export TASK_NAME=classification

CURRENT_DIR=${PWD##*/}

#    --model_type bert \
#    --model_name_or_path bert-base-uncased \

#    --model_type roberta \
#    --model_name_or_path roberta-base \

#    --model_type distilbert \
#    --model_name_or_path distilbert-base-uncased \

#    --model_type xlnet \
#    --model_name_or_path xlnet-base-cased \

#    --model_type xlm \
#    --model_name_or_path xlm-mlm-en-2048 \

#    --fp16 \
#    --fp16_opt_level O1
#    --do_lower_case \

OUT_DIR=${TASK_NAME}_roberta_7_example

echo $PWD
#python ./examples/run_toxic.py \
python -m torch.distributed.launch --nproc_per_node 2 ./examples/run_toxic.py \
    --overwrite_output_dir \
    --model_type roberta \
    --model_name_or_path roberta-base \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --data_dir $TOXIC_DATA_DIR \
    --max_seq_length 256 \
    --per_gpu_eval_batch_size=32 \
    --per_gpu_train_batch_size=32 \
    --learning_rate 5e-6 \
    --adam_epsilon 1e-6 \
    --weight_decay 0.01 \
    --num_train_epochs 6.0 \
    --weight_strategy per_class \
    --mu 1.0 \
    --smoothness 0.001 \
    --output_dir ./$OUT_DIR/