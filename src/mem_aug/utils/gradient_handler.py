"""
Gradient handling utilities for external memory training.
Addresses gradient flow issues specific to external memory components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging
import numpy as np


class GradientHandler:
    """Handles gradient-related issues in external memory training."""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Gradient tracking
        self.gradient_stats = {}
        self.step_count = 0
        
        # Register hooks for gradient monitoring
        self._register_gradient_hooks()
        
        # Gradient clipping parameters
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        self.grad_clip_type = config.get('grad_clip_type', 'norm')  # 'norm' or 'value'
        
        # Gradient scaling for external memory components
        self.external_memory_grad_scale = config.get('external_memory_grad_scale', 1.0)
        
        # Gradient accumulation buffer for external memory
        self.external_memory_grad_buffer = {}
        
        # Detection parameters for gradient issues
        self.nan_detection = True
        self.zero_grad_detection = True
        self.exploding_grad_threshold = config.get('exploding_grad_threshold', 10.0)
        
    def _register_gradient_hooks(self):
        """Register backward hooks to monitor gradients."""
        def gradient_hook(name):
            def hook(grad):
                if grad is not None:
                    self._track_gradient_stats(name, grad)
                    grad = self._handle_gradient_issues(name, grad)
                return grad
            return hook
        
        # Register hooks for external memory parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'external_memory' in name.lower():
                param.register_hook(gradient_hook(name))
    
    def _track_gradient_stats(self, name: str, grad: torch.Tensor):
        """Track gradient statistics for debugging."""
        if name not in self.gradient_stats:
            self.gradient_stats[name] = {
                'norm_history': [],
                'max_history': [],
                'min_history': [],
                'nan_count': 0,
                'zero_count': 0
            }
        
        stats = self.gradient_stats[name]
        
        # Calculate statistics
        grad_norm = grad.norm().item()
        grad_max = grad.max().item()
        grad_min = grad.min().item()
        
        stats['norm_history'].append(grad_norm)
        stats['max_history'].append(grad_max)
        stats['min_history'].append(grad_min)
        
        # Keep only last 100 steps
        for key in ['norm_history', 'max_history', 'min_history']:
            if len(stats[key]) > 100:
                stats[key] = stats[key][-100:]
        
        # Check for issues
        if torch.isnan(grad).any():
            stats['nan_count'] += 1
            self.logger.warning(f"NaN gradients detected in {name}")
        
        if grad_norm == 0.0:
            stats['zero_count'] += 1
            self.logger.warning(f"Zero gradients detected in {name}")
    
    def _handle_gradient_issues(self, name: str, grad: torch.Tensor) -> torch.Tensor:
        """Handle common gradient issues."""
        original_grad = grad.clone()
        
        # Handle NaN gradients
        if self.nan_detection and torch.isnan(grad).any():
            self.logger.warning(f"Replacing NaN gradients in {name}")
            grad = torch.where(torch.isnan(grad), torch.zeros_like(grad), grad)
        
        # Handle exploding gradients
        grad_norm = grad.norm().item()
        if grad_norm > self.exploding_grad_threshold:
            self.logger.warning(f"Clipping exploding gradients in {name}: {grad_norm:.4f}")
            grad = grad * (self.exploding_grad_threshold / grad_norm)
        
        # Apply external memory specific scaling
        if 'external_memory' in name.lower():
            grad = grad * self.external_memory_grad_scale
        
        return grad
    
    def clip_gradients(self, parameters: List[nn.Parameter]) -> float:
        """Clip gradients with improved handling for external memory."""
        # Separate external memory and other parameters
        external_memory_params = []
        other_params = []
        
        for param in parameters:
            if param.grad is not None:
                # Find parameter name
                param_name = None
                for name, p in self.model.named_parameters():
                    if p is param:
                        param_name = name
                        break
                
                if param_name and 'external_memory' in param_name.lower():
                    external_memory_params.append(param)
                else:
                    other_params.append(param)
        
        total_norm = 0.0
        
        # Clip external memory gradients with different threshold
        if external_memory_params:
            ext_memory_norm = torch.nn.utils.clip_grad_norm_(
                external_memory_params, 
                max_norm=self.max_grad_norm * 2.0  # More lenient for external memory
            )
            total_norm += ext_memory_norm ** 2
        
        # Clip other gradients normally
        if other_params:
            other_norm = torch.nn.utils.clip_grad_norm_(
                other_params, 
                max_norm=self.max_grad_norm
            )
            total_norm += other_norm ** 2
        
        total_norm = total_norm ** 0.5
        
        return total_norm
    
    def accumulate_external_memory_gradients(self, accumulation_steps: int):
        """Special handling for external memory gradient accumulation."""
        # Store external memory gradients for accumulation
        for name, param in self.model.named_parameters():
            if param.grad is not None and 'external_memory' in name.lower():
                if name not in self.external_memory_grad_buffer:
                    self.external_memory_grad_buffer[name] = torch.zeros_like(param.grad)
                
                # Accumulate gradients
                self.external_memory_grad_buffer[name] += param.grad / accumulation_steps
        
        # Replace gradients with accumulated ones at the end of accumulation
        if self.step_count % accumulation_steps == 0:
            for name, param in self.model.named_parameters():
                if name in self.external_memory_grad_buffer:
                    param.grad = self.external_memory_grad_buffer[name].clone()
                    self.external_memory_grad_buffer[name].zero_()
    
    def check_gradient_flow(self) -> Dict[str, Any]:
        """Check gradient flow and return diagnostic information."""
        gradient_info = {
            'total_parameters': 0,
            'parameters_with_gradients': 0,
            'external_memory_parameters': 0,
            'external_memory_with_gradients': 0,
            'gradient_norms': {},
            'issues': []
        }
        
        for name, param in self.model.named_parameters():
            gradient_info['total_parameters'] += 1
            
            if 'external_memory' in name.lower():
                gradient_info['external_memory_parameters'] += 1
                
                if param.grad is not None:
                    gradient_info['external_memory_with_gradients'] += 1
                    grad_norm = param.grad.norm().item()
                    gradient_info['gradient_norms'][name] = grad_norm
                    
                    # Check for issues
                    if torch.isnan(param.grad).any():
                        gradient_info['issues'].append(f"NaN gradients in {name}")
                    if grad_norm == 0.0:
                        gradient_info['issues'].append(f"Zero gradients in {name}")
                    if grad_norm > self.exploding_grad_threshold:
                        gradient_info['issues'].append(f"Exploding gradients in {name}: {grad_norm:.4f}")
                else:
                    gradient_info['issues'].append(f"No gradients for external memory parameter: {name}")
            
            elif param.grad is not None:
                gradient_info['parameters_with_gradients'] += 1
        
        return gradient_info
    
    def get_gradient_statistics(self) -> Dict[str, Any]:
        """Get comprehensive gradient statistics."""
        stats = {
            'step_count': self.step_count,
            'parameter_stats': {}
        }
        
        for name, param_stats in self.gradient_stats.items():
            if param_stats['norm_history']:
                recent_norms = param_stats['norm_history'][-10:]  # Last 10 steps
                stats['parameter_stats'][name] = {
                    'mean_norm': np.mean(recent_norms),
                    'std_norm': np.std(recent_norms),
                    'max_norm': np.max(recent_norms),
                    'min_norm': np.min(recent_norms),
                    'nan_count': param_stats['nan_count'],
                    'zero_count': param_stats['zero_count']
                }
        
        return stats
    
    def log_gradient_info(self, step: int):
        """Log gradient information for debugging."""
        if step % 100 == 0:  # Log every 100 steps
            gradient_info = self.check_gradient_flow()
            
            self.logger.info(
                f"Step {step} - Gradient Info: "
                f"External memory params with grads: {gradient_info['external_memory_with_gradients']}/{gradient_info['external_memory_parameters']}, "
                f"Total params with grads: {gradient_info['parameters_with_gradients']}/{gradient_info['total_parameters']}"
            )
            
            if gradient_info['issues']:
                self.logger.warning(f"Gradient issues: {gradient_info['issues']}")
    
    def update_step(self):
        """Update step counter."""
        self.step_count += 1
    
    def reset_buffers(self):
        """Reset gradient accumulation buffers."""
        for name in self.external_memory_grad_buffer:
            self.external_memory_grad_buffer[name].zero_()


class ExternalMemoryGradientOptimizer:
    """Specialized optimizer for external memory components."""
    
    def __init__(self, model: nn.Module, base_optimizer: torch.optim.Optimizer, config: Dict[str, Any]):
        self.model = model
        self.base_optimizer = base_optimizer
        self.config = config
        self.gradient_handler = GradientHandler(model, config)
        
        # External memory specific settings
        self.external_memory_momentum = config.get('external_memory_momentum', 0.9)
        self.external_memory_weight_decay = config.get('external_memory_weight_decay', 0.01)
        
        # Adaptive learning rate for external memory
        self.adaptive_lr = config.get('adaptive_lr', False)
        self.lr_adaptation_window = config.get('lr_adaptation_window', 100)
        
        # Performance tracking
        self.performance_history = []
    
    def step(self, closure=None):
        """Perform optimization step with gradient handling."""
        # Check gradient flow before optimization
        gradient_info = self.gradient_handler.check_gradient_flow()
        
        # Clip gradients
        trainable_params = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
        grad_norm = self.gradient_handler.clip_gradients(trainable_params)
        
        # Perform optimization step
        result = self.base_optimizer.step(closure)
        
        # Update gradient handler
        self.gradient_handler.update_step()
        
        # Log gradient information
        self.gradient_handler.log_gradient_info(self.gradient_handler.step_count)
        
        return result, grad_norm, gradient_info
    
    def zero_grad(self):
        """Zero gradients with buffer reset."""
        self.base_optimizer.zero_grad()
        self.gradient_handler.reset_buffers()
    
    def state_dict(self):
        """Get optimizer state dict."""
        return self.base_optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load optimizer state dict."""
        return self.base_optimizer.load_state_dict(state_dict)


def create_gradient_aware_optimizer(model: nn.Module, config: Dict[str, Any]) -> ExternalMemoryGradientOptimizer:
    """Create a gradient-aware optimizer for external memory training."""
    
    # Separate parameters by type
    external_memory_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if any(keyword in name.lower() for keyword in [
                'external_memory', 'external_mem_proj', 'gate_network', 'g_retrieve'
            ]):
                external_memory_params.append(param)
            else:
                other_params.append(param)
    
    # Create parameter groups with different settings
    param_groups = []
    
    if external_memory_params:
        param_groups.append({
            'params': external_memory_params,
            'lr': config.get('external_memory_lr', 1e-4),
            'weight_decay': config.get('external_memory_weight_decay', 0.01),
            'betas': config.get('external_memory_betas', (0.9, 0.999)),
            'eps': config.get('external_memory_eps', 1e-8)
        })
    
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': config.get('learning_rate', 5e-5),
            'weight_decay': config.get('weight_decay', 0.01),
            'betas': config.get('betas', (0.9, 0.999)),
            'eps': config.get('eps', 1e-8)
        })
    
    # Create base optimizer
    base_optimizer = torch.optim.AdamW(param_groups)
    
    # Wrap with gradient-aware optimizer
    return ExternalMemoryGradientOptimizer(model, base_optimizer, config)


def diagnose_gradient_issues(model: nn.Module, loss: torch.Tensor, config: Dict[str, Any]) -> Dict[str, Any]:
    """Diagnose gradient issues after backward pass."""
    diagnostics = {
        'loss_value': loss.item(),
        'loss_finite': torch.isfinite(loss).item(),
        'parameter_diagnostics': {},
        'recommendations': []
    }
    
    # Check each parameter
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_diag = {
                'has_gradient': param.grad is not None,
                'parameter_finite': torch.isfinite(param).all().item(),
            }
            
            if param.grad is not None:
                param_diag.update({
                    'gradient_finite': torch.isfinite(param.grad).all().item(),
                    'gradient_norm': param.grad.norm().item(),
                    'gradient_max': param.grad.max().item(),
                    'gradient_min': param.grad.min().item(),
                    'gradient_zero': (param.grad == 0).all().item()
                })
                
                # Check for issues and recommendations
                if not param_diag['gradient_finite']:
                    diagnostics['recommendations'].append(f"Non-finite gradients in {name}")
                
                if param_diag['gradient_norm'] > 10.0:
                    diagnostics['recommendations'].append(f"Large gradient norm in {name}: {param_diag['gradient_norm']:.4f}")
                
                if param_diag['gradient_zero']:
                    diagnostics['recommendations'].append(f"Zero gradients in {name}")
            
            else:
                diagnostics['recommendations'].append(f"No gradients computed for {name}")
            
            diagnostics['parameter_diagnostics'][name] = param_diag
    
    return diagnostics