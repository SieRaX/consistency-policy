python train.py --config-dir=configs/ --config-name=edm_square.yaml logging.name=edm_square \
hydra.run.dir='outputs/square_hybrid_reproduction/${now:%H.%M.%S}_${name}_${task_name}_cnn_16' \
logging.group=square_hybrid_reproduction \
logging.name='${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_unet_16' \
training.device=cuda:0