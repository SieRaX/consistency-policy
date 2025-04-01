python train.py --config-dir=configs/ --config-name=edm_pusht.yaml \
hydra.run.dir='outputs/pusht_hybrid_reproduction/${now:%H.%M.%S}_${name}_${task_name}_cnn_128' \
logging.group=CP_Pusht_Hybrid_Reproduction \
logging.name='${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_cnn_128' \
training.device=cuda:1 \
horizon=128 \
policy.horizon=128 \
task.dataset.horizon=128 \
task.dataset.pad_after=127 