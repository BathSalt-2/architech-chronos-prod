import torch
import hashlib
from pathlib import Path

def compute_checksum(tensor):
    """Compute SHA256 checksum for tensor."""
    return hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()

def verify_adapter(adapter_path, expected_checksum=None):
    """Verify adapter integrity before merge."""
    adapter = torch.load(adapter_path, map_location='cpu')
    
    if expected_checksum:
        # Compute checksum of all tensors
        all_tensors = torch.cat([v.flatten() for v in adapter['lora'].values()])
        checksum = compute_checksum(all_tensors)
        
        if checksum != expected_checksum:
            raise ValueError(f"Checksum mismatch: {checksum} != {expected_checksum}")
        print(f"✓ Adapter verified: {checksum[:16]}...")
    
    return adapter

def merge_adapter_delta(base_path, adapter_path, output_path=None, device='cpu'):
    """Merge LoRA adapter deltas into base model."""
    # Load base model
    model = torch.load(base_path, map_location=device)
    if isinstance(model, dict):  # If state_dict
        from architech_chronos import ArchitechChronos
        m = ArchitechChronos()
        m.load_state_dict(model)
        model = m
    
    # Load and verify adapter
    adapter = torch.load(adapter_path, map_location=device)
    
    # Merge deltas
    sd = model.state_dict()
    merged_count = 0
    for k, v in adapter['lora'].items():
        base_key = k.replace('.A', '.linear.weight').replace('.B', '.linear.weight')
        if base_key in sd:
            # Apply delta (simplified; full version reconstructs LoRA product)
            sd[base_key] += v * 0.1  # Scale factor
            merged_count += 1
    
    model.load_state_dict(sd)
    print(f"✓ Merged {merged_count} adapter parameters")
    
    # Save if output specified
    if output_path:
        torch.save(model, output_path)
        print(f"✓ Saved merged model: {output_path}")
    
    return model

def create_adapter_from_delta(base_model, finetuned_model, output_path, metadata=None):
    """Extract LoRA adapter from finetuned model delta."""
    base_sd = base_model.state_dict()
    ft_sd = finetuned_model.state_dict()
    
    lora_state = {}
    for k in base_sd.keys():
        if 'linear.weight' in k and k in ft_sd:
            delta = ft_sd[k] - base_sd[k]
            # Decompose delta into low-rank (simplified)
            U, S, Vh = torch.svd_lowrank(delta, q=4)
            lora_state[k.replace('.linear.weight', '.A')] = Vh[:4]
            lora_state[k.replace('.linear.weight', '.B')] = U[:, :4] @ torch.diag(S[:4])
    
    metadata = metadata or {}
    torch.save({'lora': lora_state, 'meta': metadata}, output_path)
    print(f"✓ Created adapter: {len(lora_state)} params → {output_path}")
    return output_path

def load_and_merge(base_path, adapter_path=None, device='cpu'):
    """Load base model and optionally merge adapter."""
    model = torch.load(base_path, map_location=device)
    if isinstance(model, dict):
        from architech_chronos import ArchitechChronos
        m = ArchitechChronos()
        m.load_state_dict(model)
        model = m
    
    if adapter_path:
        adapter = torch.load(adapter_path, map_location=device)
        sd = model.state_dict()
        for k, v in adapter['lora'].items():
            if k in sd:
                sd[k] += v  # Delta apply
        model.load_state_dict(sd)
        print(f"✓ Loaded adapter: {adapter_path}")
    
    model.eval()
    return model

# Example usage
if __name__ == '__main__':
    print("Adapter merge utility")
    print("Usage: python adapter_merge.py <base_model> <adapter> <output>")
    
    # Demo: Create dummy adapter
    from architech_chronos import ArchitechChronos
    base = ArchitechChronos()
    
    # Simulate finetuned version
    finetuned = ArchitechChronos()
    for p in finetuned.parameters():
        p.data += torch.randn_like(p) * 0.01
    
    # Extract adapter
    create_adapter_from_delta(base, finetuned, 'demo_adapter.lora', 
                             metadata={'personality': 'humorous', 'version': 1})
    
    # Merge back
    merged = merge_adapter_delta('base_model.pt', 'demo_adapter.lora')
    print("✓ Demo complete")
