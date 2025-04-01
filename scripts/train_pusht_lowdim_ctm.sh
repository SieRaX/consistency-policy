python train.py --config-dir=configs/ --config-name=ctmp_pusht_lowdim.yaml \
hydra.run.dir='outputs/pusht_lowdim_reproduction/${now:%Y.%m.%d-%H.%M.%S}_${now:%H.%M.%S}_${name}_${task_name}_cnn_16_ctmp_condition_loss_enabled' \
policy.teacher_path='outputs/pusht_lowdim_reproduction/15.40.32_train_diffusion_unet_hybrid_pusht_image_cnn_16/checkpoints/epoch\=0300-test_mean_score\=0.890.ckpt' \
policy.edm='outputs/pusht_lowdim_reproduction/15.40.32_train_diffusion_unet_hybrid_pusht_image_cnn_16/checkpoints/epoch\=0300-test_mean_score\=0.890.ckpt' \
logging.group=pusht_lowdim_reproduction \
logging.name='${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_unet_16_ctmp_condition_loss_enabled' \
training.device=cuda:0

# --config-dir=configs/ --config-name=ctmp_pusht_lowdim.yaml hydra.run.dir='outputs/pusht_lowdim_reproduction/debug/${now:%Y.%m.%d-%H.%M.%S}_${now:%H.%M.%S}_${name}_${task_name}_cnn_16_ctmp' policy.teacher_path='outputs/pusht_lowdim_reproduction/15.40.32_train_diffusion_unet_hybrid_pusht_image_cnn_16/checkpoints/epoch\=0300-test_mean_score\=0.890.ckpt' policy.edm='outputs/pusht_lowdim_reproduction/15.40.32_train_diffusion_unet_hybrid_pusht_image_cnn_16/checkpoints/epoch\=0300-test_mean_score\=0.890.ckpt' logging.group=pusht_lowdim_reproduction logging.name='Debug' training.device=cuda:1 training.debug=true