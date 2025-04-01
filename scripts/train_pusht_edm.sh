python train.py --config-dir=configs/ --config-name=edm_pusht.yaml  \
hydra.run.dir='outputs/pusht_hybrid_reproduction/${now:%Y.%m.%d-%H.%M.%S}_${now:%H.%M.%S}_${name}_${task_name}_cnn_16' \
logging.group=pusht_hybrid_reproduction \
logging.name='${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_unet_16' \
training.device=cuda:1