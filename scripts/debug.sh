python train.py --config-dir=configs/ --config-name=ctmp_pusht_lowdim.yaml \
hydra.run.dir='outputs/pusht_lowdim_reproduction/debug/${now:%Y.%m.%d-%H.%M.%S}_${now:%H.%M.%S}_${name}_${task_name}_cnn_16_ctmp' \
policy.teacher_path='outputs/pusht_lowdim_reproduction/15.40.32_train_diffusion_unet_hybrid_pusht_image_cnn_16/checkpoints/epoch\=0300-test_mean_score\=0.890.ckpt' \
policy.edm='outputs/pusht_lowdim_reproduction/15.40.32_train_diffusion_unet_hybrid_pusht_image_cnn_16/checkpoints/epoch\=0300-test_mean_score\=0.890.ckpt' \
logging.group=Debug \
logging.name='${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_unet_16_ctmp' \
training.device=cuda:0 \
training.debug=true

# python train.py --config-dir=configs/ --config-name=edm_pusht_lowdim.yaml \
# hydra.run.dir='outputs/pusht_lowdim_reproduction/debug/${now:%H.%M.%S}_${name}_${task_name}_cnn_64' \
# logging.group=Debug \
# logging.name='${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_unet_64' \
# training.device=cuda:1 \
# training.resume=true \
# training.resume_path='outputs/pusht_lowdim_reproduction/15.40.32_train_diffusion_unet_hybrid_pusht_image_cnn_16/checkpoints/epoch\=0300-test_mean_score\=0.890.ckpt' \
# training.debug=true


