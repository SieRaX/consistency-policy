# python eval.py \
# --checkpoint outputs/pusht_lowdim_reproduction/2025.04.01-14.37.00_14.37.00_train_diffusion_unet_hybrid_pusht_lowdim_cnn_64_obs_as_global_ctmp/checkpoints/epoch=0400-test_mean_score=0.739.ckpt \
# --output_dir outputs/pusht_lowdim_reproduction/2025.04.01-14.37.00_14.37.00_train_diffusion_unet_hybrid_pusht_lowdim_cnn_64_obs_as_global_ctmp/eval_gradient_32_10_batch \
# --n_action_steps 32 \
# --n_test_vis 58 \
# --device cuda:0

# python eval.py \
# --checkpoint outputs/pusht_lowdim_reproduction/2025.03.30-13.16.43_13.16.43_train_diffusion_unet_hybrid_pusht_lowdim_cnn_16_obs_as_global_ctmp/checkpoints/epoch=0250-test_mean_score=0.777.ckpt \
# --output_dir outputs/pusht_lowdim_reproduction/2025.03.30-13.16.43_13.16.43_train_diffusion_unet_hybrid_pusht_lowdim_cnn_16_obs_as_global_ctmp/eval_gradient_horizon_8_50_epi \
# --n_action_steps 8 \
# --n_test_vis 58 \
# --device cuda:0

# python eval.py \
# --checkpoint outputs/pusht_lowdim_reproduction/2025.03.30-13.16.43_13.16.43_train_diffusion_unet_hybrid_pusht_lowdim_cnn_16_obs_as_global_ctmp/checkpoints/epoch=0250-test_mean_score=0.777.ckpt \
# --output_dir outputs/pusht_lowdim_reproduction/2025.03.30-13.16.43_13.16.43_train_diffusion_unet_hybrid_pusht_lowdim_cnn_16_obs_as_global_ctmp/eval_gradient_horizon_8_50epi_gpu_100batch_plot_action \
#  --n_action_steps 8 \
#  --n_test_vis 58 \
#  --device cuda:0

python eval.py \
--checkpoint outputs/square_lowdim_reproduction/12.20.56_train_diffusion_unet_hybrid_square_lowdim_cnn_16_ctm/checkpoints/epoch=0400-test_mean_score=0.920.ckpt \
--output_dir outputs/square_lowdim_reproduction/12.20.56_train_diffusion_unet_hybrid_square_lowdim_cnn_16_ctm/eval_gradient_horizon_8_plot_each_step \
--n_test_vis 58 \
--device cuda:1
# --checkpoint outputs/square_lowdim_reproduction/12.20.56_train_diffusion_unet_hybrid_square_lowdim_cnn_16_ctm/checkpoints/epoch=0400-test_mean_score=0.920.ckpt --output_dir outputs/square_lowdim_reproduction/12.20.56_train_diffusion_unet_hybrid_square_lowdim_cnn_16_ctm/eval_gradient_horizon_8 --device cuda:1

# --checkpoint outputs/pusht_lowdim_reproduction/2025.03.30-13.16.43_13.16.43_train_diffusion_unet_hybrid_pusht_lowdim_cnn_16_obs_as_global_ctmp/checkpoints/epoch=0250-test_mean_score=0.777.ckpt --output_dir outputs/pusht_lowdim_reproduction/2025.03.30-13.16.43_13.16.43_train_diffusion_unet_hybrid_pusht_lowdim_cnn_16_obs_as_global_ctmp/eval/debug --n_action_steps 8 --n_test_vis 10 --device cuda:0