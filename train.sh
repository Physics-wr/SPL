source activate my_env
export FASTREID_DATASETS=/ai/GROUP/zhangsan/datasets
python3 train.py --config-file /ai/GROUP/zhangsan/domainReid02/configs/sam_reid.yml --num-gpus 4   OUTPUT_DIR logs/