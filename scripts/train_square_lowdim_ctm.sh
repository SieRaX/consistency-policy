python train.py --config-dir=configs/ --config-name=ctmp_square_lowdim.yaml \
hydra.run.dir='outputs/square_lowdim_reproduction/${now:%H.%M.%S}_${name}_${task_name}_cnn_16_ctm' \
policy.teacher_path='outputs/square_lowdim_reproduction/10.13.57_train_diffusion_unet_hybrid_square_lowdim_cnn_16/checkpoints/epoch\=0150-test_mean_score\=0.920.ckpt' \
policy.edm='outputs/square_lowdim_reproduction/10.13.57_train_diffusion_unet_hybrid_square_lowdim_cnn_16/checkpoints/epoch\=0150-test_mean_score\=0.920.ckpt' \
logging.name='${now:%H.%M.%S}_${name}_${task_name}_cnn_16_ctm' \
logging.group=square_lowdim_reproduction \
training.device=cuda:1

task_config_name=tool_hang
teacher_path=outputs/tool_hang_lowdim_ph_reproduction/train_by_seed_edm/seed_0_2025.07.29-10.21.57_train_diffusion_unet_hybrid_tool_hang_lowdim_cnn_32/checkpoints/epoch\\=0001-test_mean_score\\=0.000.ckpt
edm_path=outputs/tool_hang_lowdim_ph_reproduction/train_by_seed_edm/seed_0_2025.07.29-10.21.57_train_diffusion_unet_hybrid_tool_hang_lowdim_cnn_32/checkpoints/epoch\\=0001-test_mean_score\\=0.000.ckpt
debug=true
for seed in 0; do
    echo -e "\033[32m[Training ctm_${task_config_name}_lowdim with seed: ${seed}]\033[0m"
    python train.py --config-dir=configs/ --config-name=ctmp_${task_config_name}_lowdim.yaml hydra.run.dir='outputs/${task.name}_${task.dataset_type}_reproduction/train_by_seed_ctm/seed_${training.seed}_${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_cnn_${horizon}' logging.group='${task.name}_${task.dataset_type}_ctm' logging.name='seed_${training.seed}_ctm_${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_${horizon}' horizon=32 task.dataset.horizon=32 task.dataset.pad_after=31 training.seed=$seed training.device=cuda:0 obs_as_global_cond=True training.num_epochs=2000 policy.horizon=32 training.debug=$debug policy.teacher_path=$teacher_path policy.edm=$edm_path
done