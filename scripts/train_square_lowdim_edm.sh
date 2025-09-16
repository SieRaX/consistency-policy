python train.py --config-dir=configs/ --config-name=edm_square_lowdim.yaml \
hydra.run.dir='outputs/square_lowdim_reproduction/${now:%H.%M.%S}_${name}_${task_name}_cnn_16' \
logging.name=edm_square_lowdim \
logging.group=square_lowdim_reproduction \
training.device=cuda:0

# Check if there is debug mode...
export HYDRA_FULL_ERROR=1
export MUJOCO_GL=osmesa
task_config_name=tool_hang
debug=false
device=cuda:0
if [ "$debug" = true ]; then
    subdir="debug/"
    seed_list="0"
    logging_name="debug"
    wandb offline
else
    subdir=""
    seed_list=(2)
    logging_name='seed_${training.seed}_edm_${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_${horizon}'
    wandb online
fi
run_dir='outputs_HDD/${task.name}_${task.dataset_type}_reproduction/train_by_seed_edm/'$subdir'seed_${training.seed}_${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_cnn_${horizon}'
echo $run_dir
for seed in $seed_list; do
    echo -e "\033[32m[Training ${task_config_name} with seed: ${seed}]\033[0m"
    python train.py  --config-dir=configs/ --config-name=edm_${task_config_name}_lowdim.yaml hydra.run.dir=$run_dir logging.group='${task.name}_${task.dataset_type}_edm' logging.name=$logging_name  horizon=32 task.dataset.horizon=32 task.dataset.pad_after=31 training.seed=$seed training.device=$device obs_as_global_cond=True training.num_epochs=2000 policy.horizon=32 training.debug=$debug task.env_runner.n_envs=28
done

python train.py --config-dir=configs/ --config-name=edm_lift_lowdim.yaml hydra.run.dir='outputs/lift_lowdim_reproduction/train_by_seed_edm/debug/seed_${training.seed}_edm_${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}_${horizon}' logging.group=lift_lowdim_edm logging.name=debug horizon=32 task.dataset.horizon=32 task.dataset.pad_after=31 training.seed=0 training.device=cuda:0 obs_as_global_cond=True training.num_epochs=2000 policy.horizon=32 training.debug=true task.env_runner.n_envs=28