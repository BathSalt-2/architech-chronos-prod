# Architech-Chronos Quick Start Guide

## Installation

### 1. Clone or Download
```bash
cd architech-chronos-prod
```

### 2. Setup Environment
```bash
# Run setup script
./setup.sh

# Or manual installation
pip install -r requirements.txt
```

## Basic Usage

### Option 1: Run Demo Chat (Recommended for First-Time Users)

```bash
python examples/demo_chat.py
```

This launches an interactive chat demo with:
- Temporal memory across conversation turns
- Adjustable personality (humor, ethical stance)
- Real-time ethical drift monitoring
- Memory reset and audit reports

**Demo Commands**:
```
/humor 0.8          # Set humor level (0-1)
/stance bold        # Set ethical stance (neutral/cautious/bold)
/reset              # Clear temporal memory
/audit              # Show ethical audit report
/quit               # Exit
```

### Option 2: Export Model for Mobile

```bash
# Export to TorchScript (default, ~140MB)
python quantize_and_export.py --export ts

# Export to ONNX
python quantize_and_export.py --export onnx

# Export both formats
python quantize_and_export.py --export both

# Skip quantization (larger file, faster inference)
python quantize_and_export.py --export ts --no-quant
```

**Output**: `architech_chronos_ts.pt` (TorchScript) or `architech_chronos.onnx`

### Option 3: Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_model.py -v
pytest tests/test_agents.py -v
```

## Using the Model Programmatically

### Basic Inference

```python
from architech_chronos import ArchitechChronos
import torch

# Initialize model
model = ArchitechChronos(vocab_size=50257)
model.eval()

# Prepare input
input_ids = torch.randint(0, 50257, (1, 32))  # (batch, seq_len)
slots = torch.zeros(1, 89)  # Neutral personality

# Forward pass
with torch.no_grad():
    logits, drift, mem_state = model(input_ids, slots=slots)

print(f"Output shape: {logits.shape}")
print(f"Ethical drift: {drift:.6f}")
```

### With Personality Slots

```python
import torch

# Create personality vector (89-dim)
slots = torch.zeros(1, 89)

# Set humor level (index 64)
slots[0, 64] = 0.8  # High humor

# Set ethical bias (indices 81-88)
slots[0, 81:89] = torch.tensor([0.5, 0.3, 0.7, 0.2, 0.6, 0.4, 0.5, 0.3])

# Use in inference
logits, drift, mem_state = model(input_ids, slots=slots)
```

### With Ethical Monitoring

```python
from auditor import SigmaMatrixAuditor
import torch

# Initialize auditor
auditor = SigmaMatrixAuditor(threshold=0.05)

# Set baseline ethical state
baseline = torch.randn(1, 128)
auditor.set_baseline(baseline)

# During inference
current_ethic = torch.randn(1, 128)  # Extract from model
result = auditor.dmaic_cycle(current_ethic)

print(f"Drift: {result['drift']:.6f}")
print(f"Status: {result['analysis']['status']}")
print(f"Action: {result['correction']['action']}")

# Get full report
print(auditor.get_report())
```

### Multi-Agent System

```python
from deploy_stub import AgentBus

# Initialize agent bus
bus = AgentBus()

# Register custom agent
def my_agent(msg):
    prompt = msg.get('prompt', '')
    return {'response': f"Processed: {prompt}"}

bus.register('my_agent', my_agent)

# Dispatch message
response = bus.dispatch({
    'agent': 'my_agent',
    'prompt': 'Hello world'
})

print(response)

# View history
history = bus.get_history(limit=5)
```

## Working with LoRA Adapters

### Create Adapter

```python
from lora_adapter import patch_model_with_lora, save_lora_adapter
from architech_chronos import ArchitechChronos

# Create base model
model = ArchitechChronos()

# Patch with LoRA
model = patch_model_with_lora(model, r=4)

# Fine-tune model here...
# (your training code)

# Save adapter
save_lora_adapter(model, 'my_personality.lora', 
                 metadata={'personality': 'technical', 'version': 1})
```

### Load Adapter

```python
from lora_adapter import load_lora_adapter
from architech_chronos import ArchitechChronos

# Load base model
model = ArchitechChronos()

# Load and merge adapter
model, metadata = load_lora_adapter(model, 'my_personality.lora')

print(f"Loaded personality: {metadata.get('personality')}")
```

## Mobile Deployment

### iOS (via TorchScript)

```swift
import torch_mobile

// Load model
let model = try! TorchModule(fileAtPath: "architech_chronos_ts.pt")

// Prepare input
let inputTensor = torch.randint(0, 50257, [1, 32])

// Inference
let output = model.forward([inputTensor])
```

### Android (via TorchScript)

```java
import org.pytorch.Module;
import org.pytorch.Tensor;

// Load model
Module model = Module.load("architech_chronos_ts.pt");

// Prepare input
long[] shape = {1, 32};
Tensor inputTensor = Tensor.fromBlob(inputData, shape);

// Inference
Tensor output = model.forward(IValue.from(inputTensor)).toTensor();
```

## Configuration

Edit `config.yaml` to customize:

```yaml
architech_chronos:
  d_model: 256              # Model dimension
  num_layers: 8             # Number of layers
  temporal_mps:
    bond_dim: 16            # MPS compression
  ethics:
    threshold: 0.05         # Drift threshold
  adapters:
    default:
      r: 4                  # LoRA rank
```

## Evaluation

Run evaluation prompts:

```python
import json

# Load eval prompts
with open('examples/eval_prompts.json', 'r') as f:
    prompts = json.load(f)

# Test persona persistence
for test in prompts['persona_persistence']:
    print(f"Testing: {test['prompt']}")
    print(f"Expected traits: {test['expected_traits']}")
    # Run inference...
```

## Troubleshooting

### Model too large
- Use `--no-quant` flag sparingly
- Reduce `d_model` or `num_layers` in config
- Increase LoRA rank `r` for adapters instead

### High ethical drift
- Lower `ethics.threshold` in config
- Calibrate baseline with representative data
- Check personality slot values

### Slow inference
- Ensure quantization is enabled
- Reduce `temporal_mps.bond_dim`
- Use TorchScript export (faster than ONNX)

### Memory issues
- Reduce `temporal_mps.max_seq`
- Lower `memory_slots` count
- Clear memory state periodically

## Next Steps

1. **Train on your data**: Implement distillation in `quantize_and_export.py`
2. **Create custom adapters**: Fine-tune for specific personalities
3. **Integrate with apps**: Use deployment stubs for mobile/web
4. **Extend agents**: Add custom agents to the multi-agent bus
5. **Optimize**: Profile and tune for your target hardware

## Resources

- **Architecture docs**: See `ARCHITECTURE.md`
- **API reference**: Docstrings in source files
- **Eval prompts**: `examples/eval_prompts.json`
- **HQCI-QSCE paper**: Original research by Dustin Groves

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Submit a PR with improvements
- Contact: [your contact info]

---

**Happy hacking! 🎸⚡**
