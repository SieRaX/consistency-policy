python train.py --config-dir=configs/ --config-name=ctmp_square.yaml \
hydra.run.dir='outputs/square_hybrid_reproduction/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}_cnn_16_ctmp' \
policy.teacher_path='outputs/square_hybrid_reproduction/10.55.14_train_diffusion_unet_hybrid_square_image_cnn_16/checkpoints/epoch\=0350-test_mean_score\=0.900.ckpt' \
policy.edm='outputs/square_hybrid_reproduction/10.55.14_train_diffusion_unet_hybrid_square_image_cnn_16/checkpoints/epoch\=0350-test_mean_score\=0.900.ckpt' \
logging.group=square_hybrid_reproduction \
logging.name='${now:%Y.%m.%d}-${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_unet_16_ctmp' \
training.device=cuda:1

--config-dir=configs/ --config-name=ctmp_square.yaml hydra.run.dir='outputs/square_hybrid_reproduction/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}_cnn_16_ctmp' policy.teacher_path='outputs/square_hybrid_reproduction/10.55.14_train_diffusion_unet_hybrid_square_image_cnn_16/checkpoints/epoch\=0350-test_mean_score\=0.900.ckpt' policy.edm='outputs/square_hybrid_reproduction/10.55.14_train_diffusion_unet_hybrid_square_image_cnn_16/checkpoints/epoch\=0350-test_mean_score\=0.900.ckpt' logging.group=square_hybrid_reproduction logging.name='${now:%Y.%m.%d}-${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_unet_16_ctmp' training.device=cuda:1 training.debug=true