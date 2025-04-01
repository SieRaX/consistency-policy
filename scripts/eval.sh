python eval.py \
--checkpoint outputs/pusht_lowdim_reproduction/2025.03.30-13.16.43_13.16.43_train_diffusion_unet_hybrid_pusht_lowdim_cnn_16_obs_as_global_ctmp/checkpoints/epoch=0250-test_mean_score=0.777.ckpt \
--output_dir outputs/pusht_lowdim_reproduction/2025.03.30-13.16.43_13.16.43_train_diffusion_unet_hybrid_pusht_lowdim_cnn_16_obs_as_global_ctmp/eval \
--n_action_steps 8 \
--n_test_vis 50 \
--device cuda:1

# --checkpoint outputs/pusht_lowdim_reproduction/2025.03.30-13.16.43_13.16.43_train_diffusion_unet_hybrid_pusht_lowdim_cnn_16_obs_as_global_ctmp/checkpoints/epoch=0250-test_mean_score=0.777.ckpt --output_dir outputs/pusht_lowdim_reproduction/2025.03.30-13.16.43_13.16.43_train_diffusion_unet_hybrid_pusht_lowdim_cnn_16_obs_as_global_ctmp/eval --n_action_steps 8 --n_test_vis 50 --device cuda:1