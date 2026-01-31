import torch
from torch.autograd import grad

def loss_fn(model, x, condition, marginal_prob_std, eps=1e-3, augment_scale=1):
	"""The loss function for training score-based generative models.

	Args:
		model: A PyTorch model instance that represents a 
			time-dependent score-based model.
		x: A mini-batch of training data.    
		marginal_prob_std: A function that gives the standard deviation of 
			the perturbation kernel.
		eps: A tolerance value for numerical stability.
		augment_scale: A scale factor for the augmentation.
	"""
	assert type(augment_scale) == int
	x = x.repeat(augment_scale, 1)
	if condition is not None:
		condition = condition.repeat(augment_scale, 1)
	random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps  
	z = torch.randn_like(x)
	std = marginal_prob_std(random_t).reshape(-1, *[1 for _ in range( x.ndim-1)])
	perturbed_x = x + z * std

	# int_time = (random_t*100).to(torch.int32)
	# score = model(perturbed_x, int_time, condition)
	score = model(perturbed_x, random_t, condition)
	loss = torch.mean(torch.sum((score * std + z)**2, dim=(1)))
	return loss

def loss_fn_fixed_time(model, x, condition, marginal_prob_std, time, augment_scale=1):
		# time = torch.max(time, torch.tensor(1e-3))
		assert type(augment_scale) == int
		x = x.repeat(augment_scale, 1)
		if condition is not None:
			condition = condition.repeat(augment_scale, 1)
		time = torch.fill(torch.zeros(x.shape[0], device=x.device), time)
		z = torch.randn_like(x)
		std = marginal_prob_std(time).reshape(-1, *[1 for _ in range( x.ndim-1)])
		perturbed_x = x + z * std
		normalized_score = model(perturbed_x, time, condition)
		loss = torch.mean(torch.sum((normalized_score * std + z)**2, dim=(1)))
		
		# issue_point = torch.load("notebook/normalized_data_batch_10.pt", weights_only=False)
		# issue_point = torch.tensor(issue_point, device=x.device).repeat(x.shape[0], 1)
		
		# slice_idx = [5, 6, 7, 8, 9, 10]
		# mask = (perturbed_x - issue_point)[:, slice_idx].norm(dim=-1) < 0.2*(len(slice_idx)**0.5)
		# filtered_x = x[mask].clone().detach()

		filtered_x = x.clone().detach()
		
		return loss, filtered_x

def spatial_attention_gradient_smoothing_loss_on_y(model, x, condition, marginal_prob_std, eps=1e-3):
	'''
	1. Get norm of spatial attention on current batch --> What about time??
	2. Take gradient and get the norm of the gradient
	'''

	condition.requires_grad_(True)
	random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
	std = marginal_prob_std(random_t)

	unconditional_score = model(x, random_t, None)
	conditional_score = model(x, random_t, condition)

	spatial_attention = conditional_score - unconditional_score

	dSA_dy = grad(spatial_attention.norm(dim=-1).sum(), condition, create_graph=True)[0]
	
	## Applying std to the gradient smoothing loss
	loss = (dSA_dy.norm(dim=-1)*std).mean()
	condition.requires_grad_(False)

	return loss

def spatial_attention_gradient_smoothing_loss_on_x(model, x, condition, marginal_prob_std, eps=1e-3):

	x.requires_grad_(True)
	random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
	std = marginal_prob_std(random_t)

	unconditional_score = model(x, random_t, None)
	conditional_score = model(x, random_t, condition)

	spatial_attention = conditional_score - unconditional_score

	dSA_dx = grad(spatial_attention.norm(dim=-1).sum(), x, create_graph=True)[0]
	
	## Applying std to the gradient smoothing loss
	loss = (dSA_dx.norm(dim=-1)*std).mean()
	x.requires_grad_(False)

	return loss