python train.py --config-dir=configs/ --config-name=ctmp_pusht_lowdim.yaml \
policy._target_=consistency_policy.student.ctm_lowdim_policy_state_estimator.CTMPPUnetHybridLowdimPolicyStateEstimator \
hydra.run.dir='outputs/pusht_lowdim_reproduction/${now:%Y.%m.%d-%H.%M.%S}_${now:%H.%M.%S}_${name}_${task_name}_cnn_16_obs_as_global_state_estimator_ctmp' \
policy.teacher_path='outputs/pusht_lowdim_reproduction/2025.03.31-01.06.36_01.06.36_train_diffusion_unet_hybrid_pusht_lowdim_cnn_16_obs_as_global_state_estimator/checkpoints/epoch\=0150-test_mean_score\=0.844.ckpt' \
policy.edm='outputs/pusht_lowdim_reproduction/2025.03.31-01.06.36_01.06.36_train_diffusion_unet_hybrid_pusht_lowdim_cnn_16_obs_as_global_state_estimator/checkpoints/epoch\=0150-test_mean_score\=0.844.ckpt' \
obs_as_global_cond=true \
policy.obs_as_global_cond=true \
logging.group=pusht_lowdim_reproduction \
logging.name='${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_unet_16_obs_as_global_state_estimator_ctmp' \
training.device=cuda:1

#--config-dir=configs/ --config-name=ctmp_pusht_lowdim.yaml policy._target_=consistency_policy.student.ctm_lowdim_policy_state_estimator.CTMPPUnetHybridLowdimPolicyStateEstimator hydra.run.dir='outputs/pusht_lowdim_reproduction/debug/${now:%Y.%m.%d-%H.%M.%S}_${now:%H.%M.%S}_${name}_${task_name}_cnn_16_obs_as_global_state_estimator_ctmp' policy.teacher_path='outputs/pusht_lowdim_reproduction/2025.03.31-01.06.36_01.06.36_train_diffusion_unet_hybrid_pusht_lowdim_cnn_16_obs_as_global_state_estimator/checkpoints/epoch\=0150-test_mean_score\=0.844.ckpt' policy.edm='outputs/pusht_lowdim_reproduction/2025.03.31-01.06.36_01.06.36_train_diffusion_unet_hybrid_pusht_lowdim_cnn_16_obs_as_global_state_estimator/checkpoints/epoch\=0150-test_mean_score\=0.844.ckpt' obs_as_global_cond=true policy.obs_as_global_cond=true logging.group=pusht_lowdim_reproduction logging.name='${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_unet_16_obs_as_global_state_estimator_ctmp' training.device=cuda:1 training.debug=true