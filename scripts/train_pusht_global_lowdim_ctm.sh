python train.py --config-dir=configs/ --config-name=ctmp_pusht_lowdim.yaml \
hydra.run.dir='outputs/pusht_lowdim_reproduction/${now:%Y.%m.%d-%H.%M.%S}_${now:%H.%M.%S}_${name}_${task_name}_cnn_16_obs_as_global_ctmp' \
policy.teacher_path='outputs/pusht_lowdim_reproduction/2025.03.30-07.10.44_07.10.44_train_diffusion_unet_hybrid_pusht_lowdim_cnn_16_obs_as_global/checkpoints/epoch\=0200-test_mean_score\=0.811.ckpt' \
policy.edm='outputs/pusht_lowdim_reproduction/2025.03.30-07.10.44_07.10.44_train_diffusion_unet_hybrid_pusht_lowdim_cnn_16_obs_as_global/checkpoints/epoch\=0200-test_mean_score\=0.811.ckpt' \
obs_as_global_cond=true \
policy.obs_as_global_cond=true \
logging.group=pusht_lowdim_reproduction \
logging.name='${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_unet_16_obs_as_global_ctm' \
training.device=cuda:0
# --config-dir=configs/ --config-name=edm_pusht_lowdim.yaml hydra.run.dir='outputs/pusht_lowdim_reproduction/Debug/${now:%Y.%m.%d-%H.%M.%S}_${now:%H.%M.%S}_${name}_${task_name}_cnn_16_obs_as_global' obs_as_global_cond=true policy.obs_as_global_cond=true logging.group=pusht_lowdim_reproduction logging.name='${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_unet_16_obs_as_global' training.device=cuda:0 training.debug=true