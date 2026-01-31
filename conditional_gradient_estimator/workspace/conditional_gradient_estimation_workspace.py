import os
import random, functools
import numpy as np
import torch
import hydra
import wandb
import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from omegaconf import OmegaConf
from torch.linalg import matrix_norm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.func import jacrev, vmap

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.dataset.robomimic_replay_lowdim_dataset import RobomimicReplayLowdimDataset

from conditional_gradient_estimator.model.score_model import ScoreNet
from conditional_gradient_estimator.dataset.robomimic_replay_lowdim_dataset import RobomimicReplayLowdimDatasetWrapper
from conditional_gradient_estimator.util.sde import marginal_prob_std, diffusion_coeff
from conditional_gradient_estimator.util.loss_fn import loss_fn, loss_fn_fixed_time, spatial_attention_gradient_smoothing_loss_on_y, spatial_attention_gradient_smoothing_loss_on_x
from conditional_gradient_estimator.sampler.sampler import BaseSampler

OmegaConf.register_new_resolver("eval", eval, replace=True)

class ConditionalGradientEstimationWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']
    
    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.device = cfg.training.device
        
        # configure sde
        self.marginal_prob_std = functools.partial(marginal_prob_std, sigma=cfg.training.sigma, device=self.device)
        self.diffusion_coeff = functools.partial(diffusion_coeff, sigma=cfg.training.sigma, device=self.device)
        
        # configure model
        self.model: ScoreNet
        self.model = hydra.utils.instantiate(cfg.score_model, marginal_prob_std=self.marginal_prob_std)
        self.model.to(self.device)
        
        # configure train and val dataset
        dataset: RobomimicReplayLowdimDatasetWrapper
        root_dataset = hydra.utils.instantiate(cfg.task.dataset)
        dataset = hydra.utils.instantiate(cfg.wrapper_dataset, root_dataset=root_dataset)
        
        self.raw_data = {"data": list(), "condition": list()}
        for data in dataset:
            self.raw_data["data"].append(data["data"])
            self.raw_data["condition"].append(data["condition"])
            
        self.raw_data["data"] = torch.stack(self.raw_data["data"], dim=0)
        self.raw_data["condition"] = torch.stack(self.raw_data["condition"], dim=0)
            
        assert isinstance(dataset, RobomimicReplayLowdimDatasetWrapper)
        self.dataset = dataset
        self.train_dataloader = DataLoader(dataset, **cfg.dataloader)
        self.normalizer = dataset.get_normalizer()
        
        # configure optimizer and its scheduler
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())
        
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cfg.training.num_epochs*len(self.train_dataloader), eta_min=0.0)
        
        self.num_epochs = cfg.training.num_epochs
        self.sde_loss_weight = cfg.training.sde_loss_weight
        self.dsm_loss_weight = cfg.training.dsm_loss_weight
        self.dsm_fixed_time = cfg.training.dsm_fixed_time
        self.smoothing_loss_weight = cfg.training.smoothing_loss_weight
        self.augment_noise_scale = cfg.training.augment_noise_scale

        self.sde_min_time = cfg.training.sde_min_time
        
        self.sampler : BaseSampler
        self.sampler = hydra.utils.instantiate(cfg.sampler)
        self.sample_every = cfg.training.sample_every
        self.max_train_steps = cfg.training.max_train_steps
        
        if cfg.training.debug:
            self.num_epochs = 2
            self.max_train_steps = 10
            self.sample_every = 1
            
        # self.cfg for saving checkpoints
        self.cfg = cfg
        
        """
        We need
        1. Model o
        2. optimizer o
        3. cosine scheduler o
        4. dataset o
        5. dataloader o
        6. epoch o
        7. ploting interval
        8. checkpoint interval
        9. save last checkpoint
        10. save last snapshot
        11. save best checkpoint
        12. save best snapshot
        13. save last checkpoint
        14. save last snapshot
        15. wandb support
        
        """

    def run(self):
        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(self.cfg, resolve=True),
            **self.cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )
        
        
        global_step = 0
        filtered_x_unconditional_list = list()        
        for epoch in range(self.num_epochs):
            
            step_log = dict()
            train_losses = list()

            
            with tqdm.tqdm(self.train_dataloader, desc=f"Training epoch {epoch}|{self.num_epochs}", leave=False) as tepoch:
                for batch_idx, data in enumerate(tepoch):
                    x = self.normalizer['data'].normalize(data['data']).to(self.device)
                    y = self.normalizer['condition'].normalize(data['condition']).to(self.device)
                    
                    augment_scale = self.augment_noise_scale
                    ## get loss
                    conditional_loss = loss_fn(self.model, x, y, self.marginal_prob_std, eps=self.sde_min_time, augment_scale=augment_scale)
                    # conditional_loss = torch.tensor(0.0, device=self.device)
                    unconditional_loss = loss_fn(self.model, x, None, self.marginal_prob_std, eps=self.sde_min_time, augment_scale=augment_scale)

                    
                    fixed_time = torch.tensor(self.dsm_fixed_time, device=self.device)
                    fixed_time_loss_unconditional, filtered_x_unconditional = loss_fn_fixed_time(self.model, x, None, self.marginal_prob_std, fixed_time, augment_scale=augment_scale)
                    filtered_x_unconditional_list.append(filtered_x_unconditional)
                    
                    fixed_time_loss_conditional, _ = loss_fn_fixed_time(self.model, x, y, self.marginal_prob_std, fixed_time, augment_scale=augment_scale)

                    # conditioned_score_norm = self.model(x, torch.fill(torch.zeros(x.shape[0], device=x.device), 0.1), y).norm(dim=-1).detach().cpu().numpy().mean()
                    
                    # # Calculate gradient smoothing loss
                    # x.requires_grad_(True)
                    # model_output = self.model(x, torch.fill(torch.zeros(x.shape[0], device=x.device), self.sde_min_time), y)
                    # grad = torch.autograd.grad(model_output.sum(), x, create_graph=True)[0]
                    # smoothing_loss_x = torch.norm(grad, p=2)
                    # x.requires_grad_(False)
                    
                    # # Calculate gradient smoothing loss
                    # y.requires_grad_(True)
                    # model_output = self.model(x, torch.fill(torch.zeros(x.shape[0], device=x.device), self.sde_min_time), y)
                    # grad = torch.autograd.grad(model_output.sum(), y, create_graph=True)[0]
                    # smoothing_loss_y = torch.norm(grad, p=2)
                    # y.requires_grad_(False)
                    
                    # # Calculate gradient smoothing loss
                    # x.requires_grad_(True)
                    # y.requires_grad_(True)
                    # # model_output = self.model(x, torch.fill(torch.zeros(x.shape[0], device=x.device), self.sde_min_time), y) - self.model(x, torch.fill(torch.zeros(x.shape[0], device=x.device), self.sde_min_time), None)
                    
                    # # Calculate Jacobian with respect to both x and y using jacrev
                    # # For batched inputs, we can use a more efficient approach
                    
                    # # Define function that takes x and y as separate arguments
                    # def model_fn(x_input, y_input):
                    #     x_input = x_input.unsqueeze(0)
                    #     y_input = y_input.unsqueeze(0)
                    #     return (self.model(x_input, torch.fill(torch.zeros(x_input.shape[0], device=x_input.device), fixed_time), y_input)).squeeze(0)
                    
                    # # # Compute Jacobians using vmap to vectorize over batch dimension
                    # def compute_jacobians(x_single, y_single):
                    #     jac_x= jacrev(model_fn, argnums=(0))(x_single, y_single)
                    #     return jac_x
                    
                    # # # Vectorize over batch dimension - returns tuple of Jacobians
                    # jacobian = vmap(compute_jacobians, in_dims=0)(x, y)
                    # # row, column = jacobian.shape[-2:]
                    
                    # x.requires_grad_(False)
                    # y.requires_grad_(False)
                    
                    # # # Calculate smoothing loss using Jacobian norms
                    # smoothing_loss = matrix_norm(jacobian).mean()
                    smoothing_loss_y = spatial_attention_gradient_smoothing_loss_on_y(self.model, x, y, self.marginal_prob_std, eps=self.sde_min_time)
                    smoothing_loss_x = spatial_attention_gradient_smoothing_loss_on_x(self.model, x, y, self.marginal_prob_std, eps=self.sde_min_time)
                    smoothing_loss = smoothing_loss_x + smoothing_loss_y
                    # smoothing_loss = torch.tensor(0.0)
                    
                    
                    # # # smoothing_loss = smoothing_loss_x + 0.01 * smoothing_loss_y # For robomimic
                    # # smoothing_loss = smoothing_loss_x + smoothing_loss_y # For D4RL
                    
                    loss = \
                        self.sde_loss_weight * (conditional_loss + unconditional_loss) + \
                        self.dsm_loss_weight * (fixed_time_loss_conditional + fixed_time_loss_unconditional) + \
                        self.smoothing_loss_weight * smoothing_loss
                    
                    ## backward
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    
                    train_losses.append(loss.item())
                    
                    step_log = {
                        'train_loss': loss.item(),
                        'sde_loss': conditional_loss.item() + unconditional_loss.item(),
                        'dsm_loss': fixed_time_loss_conditional.item() + fixed_time_loss_unconditional.item(),
                        'smoothing_loss': smoothing_loss.item(),
                        # 'conditioned_score_norm': conditioned_score_norm,
                        'global_step': global_step,
                        'epoch': epoch,
                        'lr': self.scheduler.get_last_lr()[0]
                    }
                    
                    is_last_batch = (batch_idx == (len(self.train_dataloader)-1))
                    if not is_last_batch:
                        # log of last step is combined with validation and rollout
                        wandb_run.log(step_log, step=global_step)
                        global_step += 1
                        
                    if (self.max_train_steps is not None) \
                            and batch_idx >= (self.max_train_steps-1):
                            break
                    
            train_loss = np.mean(train_losses)
            step_log['train_loss'] = train_loss
            
            ## evaluation ##
            # 1. sample unconditional data and compare with groudn truth
            # 2. sample conditional data and compare with ground truth
            
            if ((epoch+1) % self.sample_every) == 0:
                self.model.eval()
                with torch.no_grad():
                    # gt_data = data['data'].to(self.device)
                    # gt_data = self.train_dataloader.dataset[:].to(self.device)
                    
                    unconditional_samples = self.sampler.sample(
                        self.model,
                        None,
                        self.marginal_prob_std,
                        self.diffusion_coeff,
                        device=self.device,
                    )
                    # unconditional_samples = self.normalizer['data'].unnormalize(unconditional_samples)
                    
                    idx = torch.randint(0, len(self.train_dataloader.dataset), (1,))
                    data = self.train_dataloader.dataset[idx]
                    condition = self.normalizer['condition'].normalize(data['condition']).to(self.device)
                    conditional_gt = data['data'].to(self.device)
                    print(f"conditional_gt: {conditional_gt.shape}")
                    input()
                    conditional_gt = self.normalizer['data'].normalize(conditional_gt)
                    sampling_condition = condition.repeat(self.sampler.batch_size, 1).to(self.device)
                    normalized_raw_data = self.normalizer['data'].normalize(self.raw_data["data"])
                    
                    conditional_samples = self.sampler.sample(
                        self.model,
                        sampling_condition,
                        self.marginal_prob_std,
                        self.diffusion_coeff,
                        device=self.device,
                    )
                    # conditional_samples = self.normalizer['data'].unnormalize(conditional_samples)
                    
                    # gt_data_np = gt_data.cpu().numpy()
                    conditional_samples_np = conditional_samples.cpu().numpy()
                    unconditional_samples_np = unconditional_samples.cpu().numpy()
                    
                    # get score
                    num_score = 2
                    conditional_gt_batch = conditional_gt.repeat(num_score, 1).to(self.device)
                    noise = torch.randn_like(conditional_gt_batch) * 0.0  # Add iid Gaussian noise with std=0.1
                    conditional_gt_batch = conditional_gt_batch + noise
                    
                    timestep = torch.ones(conditional_gt_batch.shape[0], device=conditional_gt_batch.device)*fixed_time
                    
                    condition_batch = condition.repeat(num_score, 1).to(self.device)
                    unconditioned_score = self.model(conditional_gt_batch, timestep, None).mean(dim=0).detach().cpu().numpy()
                    conditioned_score = self.model(conditional_gt_batch, timestep, condition_batch).mean(dim=0).detach().cpu().numpy()
                    # change to numpy
                    conditional_gt_batch = conditional_gt_batch.mean(dim=0).cpu().numpy()
                    
                # Plot in wandb
                tqdm_bar = tqdm.tqdm(range(self.raw_data["data"].shape[1]-1), leave=False, desc=f"Plotting")
                max_digit = len(str(self.raw_data["data"].shape[1]-1))
                for i in tqdm_bar:
                    
                    gt_scatter_data = normalized_raw_data[:, [i, i+1]]
                    # gt_scatter_data = self.raw_data["data"][:, [i, i+1]]
                    conditional_scatter_data = conditional_samples_np[:, [i, i+1]]
                    unconditional_scatter_data = unconditional_samples_np[:, [i, i+1]]
                    conditional_gt_data = conditional_gt.cpu().numpy()[[i, i+1]]
                    
                    # Create scatter plot
                    fig, ax = plt.subplots(figsize=(8,6))
                    ax.grid(True)
                    ax.scatter(gt_scatter_data[:,0], gt_scatter_data[:,1], alpha=0.5, s=2)
                    ax.scatter(conditional_scatter_data[:,0], conditional_scatter_data[:,1], alpha=0.5, s=2)
                    ax.scatter(conditional_gt_data[0], conditional_gt_data[1], marker='*', color='blue', s=100)
                    
                    # Add quiver plot for gradient information
                    if i < conditioned_score.shape[0] - 1:  # Ensure we have enough dimensions
                        # Get the gradient components for the current dimensions
                        gradient_x = conditioned_score[i]
                        gradient_y = conditioned_score[i+1]
                        unconditioned_gradient_x = unconditioned_score[i]
                        unconditioned_gradient_y = unconditioned_score[i+1]
                        
                        # Create quiver plot
                        ax.quiver(conditional_gt_batch[i], 
                                 conditional_gt_batch[i+1],
                                 gradient_x, gradient_y, 
                                 alpha=1.0, color='red', scale=1, width=0.002,
                                 label='Conditional Gradient')
                        ax.quiver(conditional_gt_batch[i], 
                                 conditional_gt_batch[i+1],
                                 unconditioned_gradient_x, unconditioned_gradient_y, 
                                 alpha=1.0, color='green', scale=1, width=0.002,
                                 label='Unconditional Gradient')
                    
                    ax.set_xlabel(f'x_{i}')
                    ax.set_ylabel(f'x_{i+1}')
                    ax.set_xlim(conditional_gt_data[0]-0.5, conditional_gt_data[0]+0.5)
                    ax.set_ylim(conditional_gt_data[1]-0.5, conditional_gt_data[1]+0.5)
                    ax.legend()
                    
                    # Convert plot to PIL Image
                    fig.canvas.draw()
                    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    conditional_scatter_img = Image.fromarray(data)
                    plt.close()
                    
                    step_log[f"conditional_scatter/{str(i).zfill(max_digit)}_{str(i+1).zfill(max_digit)}"] = wandb.Image(conditional_scatter_img)
                    
                    fig, ax = plt.subplots(figsize=(4,3))
                    ax.grid(True)
                    ax.scatter(gt_scatter_data[:,0], gt_scatter_data[:,1], alpha=0.5, label='Ground Truth', s=2)
                    ax.scatter(unconditional_scatter_data[:,0], unconditional_scatter_data[:,1], alpha=0.5, label='Unconditional', s=2)
                    ax.set_xlabel(f'x_{i}')
                    ax.set_ylabel(f'x_{i+1}')
                    ax.legend()
                    
                    # Convert plot to PIL Image
                    fig.canvas.draw()
                    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    unconditional_scatter_img = Image.fromarray(data)
                    plt.close()
                    step_log[f"unconditional_scatter/{str(i).zfill(max_digit)}_{str(i+1).zfill(max_digit)}"] = wandb.Image(unconditional_scatter_img)
                
                self.model.train()
                # save checkpoint
                self.save_checkpoint()
            
            wandb_run.log(step_log, step=global_step)
            global_step += 1
        
        filtered_x_unconditional_list = torch.cat(filtered_x_unconditional_list, dim=0)
        torch.save(filtered_x_unconditional_list, os.path.join(self.output_dir, "filtered_x_unconditional.pt"))
        # save checkpoint
        self.save_checkpoint()
            