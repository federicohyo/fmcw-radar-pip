import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

def human_readable_count(number):
    for unit in ['', 'K', 'M', 'G', 'T']:
        if abs(number) < 1000:
            return f"{number:3.1f}{unit}"
        number /= 1000
    return f"{number:.1f}P"  # for really large numbers

def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_model_parameters_nonzero(model):

    total_params = 0
    total_zero_params = 0
    for module in model.modules():
        if hasattr(module, 'weight') and getattr(module, 'weight') is not None:
            # Count all parameters and zero parameters in this module
            weight = module.weight.data
            total_params += weight.nelement()
            total_zero_params += torch.sum(weight == 0).item()

    return total_params - total_zero_params

def get_prunable_layers(model):
    prunable_layers = []
    
    # Iterate through the model's named modules
    for name, module in model.named_modules():
        # Check if the layer is a convolutional or linear layer
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prunable_layers.append((name, module))
    
    return prunable_layers

def get_prunable_parameters(model):
    # prunable_parameters = []
    
    # Iterate through the model's named parameters
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):  # Example types, adjust as needed
            if hasattr(module, 'weight'):
                parameters_to_prune.append((module, 'weight'))
    
    return parameters_to_prune

def calculate_model_sparsity(model):
    """
    Calculate the overall sparsity of the pruned model by considering all
    the layers that have been pruned.
    Args:
    - model: PyTorch model that has been pruned.
    Returns:
    - overall_sparsity: The overall sparsity percentage of the model.
    """
    total_params = 0
    total_zero_params = 0
    for module in model.modules():
        if hasattr(module, 'weight') and getattr(module, 'weight') is not None:
            # Count all parameters and zero parameters in this module
            weight = module.weight.data
            total_params += weight.nelement()
            total_zero_params += torch.sum(weight == 0).item()
    
    overall_sparsity = 100. * float(total_zero_params) / float(total_params)
    return overall_sparsity

def prune_model(pruning_config, model, model_name = 'mobilenetv2'):

    # Retrieve the pruning function based on the string
    pruning_function = getattr(prune, pruning_config['pruning_type'])

    # Apply the pruning

    if pruning_config['global_pruning']:
        # Global pruning
        if 'unstructured' in pruning_config['pruning_type']:
            parameters_to_prune = get_prunable_parameters(model)            
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=pruning_config['pruning_amount']
            )
            for module, param in parameters_to_prune:
                prune.remove(module, param)

        else:
            ValueError(f"Pruning type {pruning_config['pruning_type']} not supported for structured pruning")
    else:
        for name, module in get_prunable_layers(model):
            if 'unstructured' in pruning_config['pruning_type']:
                # For unstructured pruning methods
                pruning_function(module, name="weight", amount=pruning_config['pruning_amount'])
            else:
                ValueError(f"Pruning type {pruning_config['pruning_type']} not supported for structured pruning")
            #     # For structured pruning methods
            #     pruning_function(module, name="weight", amount=pruning_config['pruning_amount'], n=2, dim=0)
    
    print(f"Pruning with {pruning_config['pruning_type']}")
    print(f"Number of parameters before pruning: {human_readable_count(count_model_parameters(model))}")

    return model

def prune_model_iterative(pruning_config, model, model_name = 'mobilenetv2', iteration = 0):
    # Retrieve the pruning function based on the string
    pruning_function = getattr(prune, pruning_config['pruning_type'])

    # Apply the pruning
    if pruning_config['global_pruning']:
        # Global pruning
        if 'unstructured' in pruning_config['pruning_type']:
            parameters_to_prune = get_prunable_parameters(model)           
            # pruning_amount = ((iteration+1) / pruning_config['itaretive_pruning_steps']) * pruning_config['pruning_amount'] 
            pruning_amount = 1 - pruning_config['pruning_amount']
            print("Pruning amount: ", pruning_amount)
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount = pruning_amount
            )

            # Calculate the sparsity after pruning  
            sparsity = calculate_model_sparsity(model)
            print(f'After pruning step {iteration+1}, sparsity is {sparsity:.2f}%')

            # # Remove the pruned parameters
            # for module, param in parameters_to_prune:
            #     prune.remove(module, param)

        else:
            ValueError(f"Pruning type {pruning_config['pruning_type']} not supported for structured pruning")
    else:
        for name, module in get_prunable_layers(model):
            if 'unstructured' in pruning_config['pruning_type']:
                # For unstructured pruning methods
                pruning_function(module, name="weight", amount=pruning_config['pruning_amount'])
            else:
                ValueError(f"Pruning type {pruning_config['pruning_type']} not supported for structured pruning")
            #     # For structured pruning methods
            #     pruning_function(module, name="weight", amount=pruning_config['pruning_amount'], n=2, dim=0)
    
    print(f"Pruning with {pruning_config['pruning_type']}")
    print(f"Number of parameters before pruning: {human_readable_count(count_model_parameters(model))}")

    return model

