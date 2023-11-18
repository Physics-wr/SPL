source activate lrk_env
export FASTREID_DATASETS=/ai/HAG/nihao/datasets
python3 train.py --config-file /ai/HAG/nihao/domainReid02/configs/sam_reid.yml --num-gpus 4   OUTPUT_DIR logs/