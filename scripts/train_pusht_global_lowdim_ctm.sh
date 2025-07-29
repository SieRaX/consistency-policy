python train.py --config-dir=configs/ --config-name=ctmp_pusht_lowdim.yaml \
hydra.run.dir='outputs/pusht_lowdim_reproduction/${now:%Y.%m.%d-%H.%M.%S}_${now:%H.%M.%S}_${name}_${task_name}_cnn_64_obs_as_global_ctmp' \
policy.teacher_path='outputs/pusht_lowdim_reproduction/2025.04.01-03.13.41_03.13.41_train_diffusion_unet_hybrid_pusht_lowdim_cnn_64_obs_as_global/checkpoints/epoch\=0250-test_mean_score\=0.771.ckpt' \
policy.edm='outputs/pusht_lowdim_reproduction/2025.04.01-03.13.41_03.13.41_train_diffusion_unet_hybrid_pusht_lowdim_cnn_64_obs_as_global/checkpoints/epoch\=0250-test_mean_score\=0.771.ckpt' \
obs_as_global_cond=true \
policy.obs_as_global_cond=true \
logging.group=pusht_lowdim_reproduction \
logging.name='${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_unet_64_obs_as_global_ctm' \
training.device=cuda:1

# --config-dir=configs/ --config-name=ctmp_pusht_lowdim.yaml hydra.run.dir='outputs/pusht_lowdim_reproduction/debug/${now:%Y.%m.%d-%H.%M.%S}_${now:%H.%M.%S}_${name}_${task_name}_cnn_64_obs_as_global_ctmp' policy.teacher_path='outputs/pusht_lowdim_reproduction/2025.04.01-03.13.41_03.13.41_train_diffusion_unet_hybrid_pusht_lowdim_cnn_64_obs_as_global/checkpoints/epoch\=0250-test_mean_score\=0.771.ckpt' policy.edm='outputs/pusht_lowdim_reproduction/2025.04.01-03.13.41_03.13.41_train_diffusion_unet_hybrid_pusht_lowdim_cnn_64_obs_as_global/checkpoints/epoch\=0250-test_mean_score\=0.771.ckpt' obs_as_global_cond=true policy.obs_as_global_cond=true logging.group=pusht_lowdim_reproduction logging.name='${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_unet_64_obs_as_global_ctm' training.device=cuda:1 training.debug=true