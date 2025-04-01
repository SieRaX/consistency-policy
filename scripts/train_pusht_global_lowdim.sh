python train.py --config-dir=configs/ --config-name=edm_pusht_lowdim.yaml \
hydra.run.dir='outputs/pusht_lowdim_reproduction/${now:%Y.%m.%d-%H.%M.%S}_${now:%H.%M.%S}_${name}_${task_name}_cnn_16_obs_as_global' \
obs_as_global_cond=true \
policy.obs_as_global_cond=true \
logging.group=pusht_lowdim_reproduction \
logging.name='${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_unet_16_obs_as_global' \
training.device=cuda:0

# --config-dir=configs/ --config-name=edm_pusht_lowdim.yaml hydra.run.dir='outputs/pusht_lowdim_reproduction/Debug/${now:%Y.%m.%d-%H.%M.%S}_${now:%H.%M.%S}_${name}_${task_name}_cnn_16_obs_as_global' obs_as_global_cond=true policy.obs_as_global_cond=true logging.group=pusht_lowdim_reproduction logging.name='${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_unet_16_obs_as_global' training.device=cuda:0 training.debug=true