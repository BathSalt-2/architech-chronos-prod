import torch
import torch.nn as nn
import yaml

with open('config.yaml', 'r') as f:
    CFG = yaml.safe_load(f)['architech_chronos']

class LoRALinear(nn.Module):
    """Drop-in LoRA for Linear layers – r=4, alpha=16 per spec."""
    def __init__(self, in_dim, out_dim, r=None, alpha=16):
        super().__init__()
        r = r or CFG['adapters']['default']['r']
        self.r = r
        self.scale = alpha / r
        self.linear = nn.Linear(in_dim, out_dim)  # Base
        if r > 0:
            self.A = nn.Parameter(torch.randn(r, in_dim) * 0.01)
            self.B = nn.Parameter(torch.zeros(out_dim, r))
        else:
            self.A = None
            self.B = None

    def forward(self, x):
        base = self.linear(x)
        if self.r > 0:
            delta = (x @ self.A.T) @ self.B.T * self.scale
            return base + delta
        return base

def patch_model_with_lora(model, target_modules=['Linear'], r=4):
    """Replace target modules with LoRA variants."""
    for name, module in model.named_modules():
        if any(t in type(module).__name__ for t in target_modules):
            if isinstance(module, nn.Linear):
                in_dim = module.in_features
                out_dim = module.out_features
                lora_layer = LoRALinear(in_dim, out_dim, r=r)
                lora_layer.linear.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    lora_layer.linear.bias.data = module.bias.data.clone()
                # Replace in parent
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = dict(model.named_modules())[parent_name] if parent_name else model
                setattr(parent, child_name, lora_layer)
    return model

def extract_lora_state(model):
    """Extract only LoRA params (A, B matrices) for OTA updates."""
    lora_state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear) and module.r > 0:
            lora_state[f"{name}.A"] = module.A.data
            lora_state[f"{name}.B"] = module.B.data
    return lora_state

def save_lora_adapter(model, path, metadata=None):
    """Save LoRA adapter with metadata (personality slot config, etc.)."""
    lora_state = extract_lora_state(model)
    metadata = metadata or {}
    torch.save({'lora': lora_state, 'meta': metadata}, path)
    print(f"Saved LoRA adapter: {len(lora_state)} params → {path}")

def load_lora_adapter(model, path):
    """Load and merge LoRA adapter into model."""
    checkpoint = torch.load(path, map_location='cpu')
    lora_state = checkpoint['lora']
    for name, param in lora_state.items():
        module_name, param_name = name.rsplit('.', 1)
        module = dict(model.named_modules())[module_name]
        if hasattr(module, param_name):
            getattr(module, param_name).data = param
    print(f"Loaded LoRA adapter from {path}")
    return model, checkpoint.get('meta', {})
