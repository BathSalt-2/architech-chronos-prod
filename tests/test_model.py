import pytest
import torch
import sys
sys.path.insert(0, '..')

from architech_chronos import (
    ArchitechChronos, 
    TemporalMPSLayer, 
    ChronosMemory, 
    SigmaEthicsModule,
    PersonalityGater
)

def test_temporal_mps_layer():
    """Test temporal MPS compression layer."""
    layer = TemporalMPSLayer(d_model=256, bond_dim=16, max_seq=128)
    x = torch.randn(2, 32, 256)  # Batch=2, Seq=32, Dim=256
    
    output = layer(x)
    
    assert output.shape == x.shape, "Output shape mismatch"
    assert not torch.isnan(output).any(), "NaN detected in output"
    print("✓ TemporalMPSLayer test passed")

def test_chronos_memory():
    """Test recursive memory module."""
    memory = ChronosMemory(d_model=256, slots=8)
    h = torch.randn(2, 32, 256)
    
    output = memory(h)
    
    assert output.shape == h.shape, "Memory output shape mismatch"
    assert not torch.isnan(output).any(), "NaN in memory output"
    print("✓ ChronosMemory test passed")

def test_sigma_ethics():
    """Test ethical drift detection."""
    ethics = SigmaEthicsModule(d_model=256, ethics_dim=128)
    h = torch.randn(2, 32, 256)
    target = torch.randn(2, 128)
    
    output, drift = ethics(h, target)
    
    assert output.shape == h.shape, "Ethics output shape mismatch"
    assert isinstance(drift, float), "Drift should be float"
    assert drift >= 0, "Drift should be non-negative"
    print(f"✓ SigmaEthicsModule test passed (drift={drift:.6f})")

def test_personality_gater():
    """Test personality slot injection."""
    gater = PersonalityGater(d_model=256, slot_dim=89)
    x = torch.randn(2, 32, 256)
    slots = torch.randn(2, 89)
    
    output = gater(x, slots)
    
    assert output.shape == x.shape, "Gater output shape mismatch"
    assert not torch.isnan(output).any(), "NaN in gater output"
    print("✓ PersonalityGater test passed")

def test_full_model_forward():
    """Test full model forward pass."""
    model = ArchitechChronos(vocab_size=50257)
    input_ids = torch.randint(0, 50257, (2, 32))
    slots = torch.randn(2, 89)
    ethic_target = torch.randn(2, 128)
    
    logits, drift, mem_state = model(input_ids, ethic_target=ethic_target, slots=slots)
    
    assert logits.shape == (2, 32, 50257), "Logits shape mismatch"
    assert isinstance(drift, float), "Drift should be float"
    assert mem_state.shape == (2, 256), "Memory state shape mismatch"
    assert not torch.isnan(logits).any(), "NaN in logits"
    print(f"✓ Full model test passed (drift={drift:.6f})")

def test_model_perplexity():
    """Test model perplexity on dummy data."""
    model = ArchitechChronos(vocab_size=50257)
    model.eval()
    
    # Generate dummy sequence
    input_ids = torch.randint(0, 50257, (1, 50))
    target_ids = torch.randint(0, 50257, (1, 50))
    
    with torch.no_grad():
        logits, _, _ = model(input_ids)
        loss = torch.nn.functional.nll_loss(
            logits.view(-1, 50257), 
            target_ids.view(-1)
        )
        perplexity = torch.exp(loss).item()
    
    assert perplexity > 0, "Perplexity should be positive"
    assert perplexity < 1e6, "Perplexity unreasonably high"
    print(f"✓ Perplexity test passed (ppl={perplexity:.2f})")

def test_model_size():
    """Test model parameter count."""
    model = ArchitechChronos(vocab_size=50257)
    param_count = sum(p.numel() for p in model.parameters())
    size_mb = param_count * 4 / 1e6  # Assuming float32
    
    print(f"Model size: {param_count/1e6:.2f}M params (~{size_mb:.1f}MB)")
    assert size_mb < 200, "Model too large for mobile target"
    print("✓ Model size test passed")

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
