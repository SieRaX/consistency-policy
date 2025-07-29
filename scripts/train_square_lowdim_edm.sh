python train.py --config-dir=configs/ --config-name=edm_square_lowdim.yaml \
hydra.run.dir='outputs/square_lowdim_reproduction/${now:%H.%M.%S}_${name}_${task_name}_cnn_16' \
logging.name=edm_square_lowdim \
logging.group=square_lowdim_reproduction \
training.device=cuda:0

# Check if there is debug mode...

for seed in 0; do
    echo -e "\033[32m[Training edm_lift_lowdim with seed: ${seed}]\033[0m"
    python train.py  --config-dir=configs/ --config-name=edm_tool_hang_lowdim.yaml hydra.run.dir='outputs/${task.name}_${task.dataset_type}_reproduction/train_by_seed_edm/seed_${training.seed}_${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_cnn_${horizon}' logging.group='${task.name}_${task.dataset_type}_edm' logging.name='seed_${training.seed}_edm_${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_${horizon}'  horizon=32 task.dataset.horizon=32 task.dataset.pad_after=31 training.seed=$seed training.device=cuda:0 obs_as_global_cond=True training.num_epochs=2000 policy.horizon=32 training.debug=true
done