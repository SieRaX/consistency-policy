python train.py --config-dir=configs/ --config-name=ctmp_pusht.yaml \
hydra.run.dir='outputs/pusht_reproduction/${now:%Y.%m.%d-%H.%M.%S}_${now:%H.%M.%S}_${name}_${task_name}_cnn_16_ctmp' \
policy.teacher_path='outputs/pusht_hybrid_reproduction/2025.03.27-16.20.39_16.20.39_train_diffusion_unet_hybrid_pusht_image_cnn_16/checkpoints/epoch\=0100-test_mean_score\=0.821.ckpt' \
policy.edm='outputs/pusht_hybrid_reproduction/2025.03.27-16.20.39_16.20.39_train_diffusion_unet_hybrid_pusht_image_cnn_16/checkpoints/epoch\=0100-test_mean_score\=0.821.ckpt' \
logging.group=pusht_hybrid_reproduction \
logging.name='${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_unet_16_ctmp' \
training.device=cuda:0

# --config-dir=configs/ --config-name=ctmp_pusht.yaml hydra.run.dir='outputs/pusht_reproduction/debug/${now:%Y.%m.%d-%H.%M.%S}_${now:%H.%M.%S}_${name}_${task_name}_cnn_16_ctmp' policy.teacher_path='outputs/pusht_hybrid_reproduction/2025.03.27-16.20.39_16.20.39_train_diffusion_unet_hybrid_pusht_image_cnn_16/checkpoints/epoch\=0100-test_mean_score\=0.821.ckpt' policy.edm='outputs/pusht_hybrid_reproduction/2025.03.27-16.20.39_16.20.39_train_diffusion_unet_hybrid_pusht_image_cnn_16/checkpoints/epoch\=0100-test_mean_score\=0.821.ckpt' logging.group=Debug logging.name='${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_unet_16_ctmp' training.device=cuda:0 training.debug=true