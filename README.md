# Architech-Chronos: Quantum-Temporal AI Sidekick – Production Edition

## Overview
Your pocket Gandalf: A multi-agent, personality-aware LLM with HQCI-QSCE-inspired tensor compression, temporal recursion, and DMAIC ethical guardrails. Targets mobile (<150MB, <200ms/token), supports LoRA personalization, and orchestrates agents (Chronos, Auditor, etc.) via a lightweight JSON bus. Built for on-device epinoetics—reasoning that evolves ethically without cloud drama.

Inspired by Dustin Groves' HQCI-QSCE (Nov 2025): Tensor-network states on NPUs, RL variance-scaling, Σ-Matrix governance.

## Quick Start
1. Clone & install: `pip install -r requirements.txt`
2. Export model: `python quantize_and_export.py --export ts` (TorchScript bundle)
3. Run demo: `python examples/demo_chat.py` – Temporal chat with humor toggle.
4. Deploy: Load `architech_chronos_ts.pt` in iOS/Android (see deploy_stub.py).

## Key Features
- **Compression**: Distill + dynamic 8-bit quant → ~140MB.
- **Temporal**: MPS layers + ChronosMemory for >512-token continuity.
- **Ethics**: Runtime DMAIC drift checks (<0.05 threshold).
- **Personalization**: LoRA adapters (~3MB) for tone/humor slots.
- **Multi-Agent**: JSON protocol; agents handoff in <50ms.
- **Exports**: TorchScript (default), ONNX, TFLite stubs.

## Eval Matrix (Run `pytest tests/`)
- Perplexity: <20 on multi-turn dialogues.
- Ethical Drift: MSE <0.01/session.
- Latency: Benchmark on A17: ~150ms/token.
- Memory Retention: 90% recall @1h.
- Human Prompts: See `examples/eval_prompts.json` – A/B persona persistence.

## Tradeoffs
- 8-bit quant: Fast, but validate lang quality (no 4-bit yet).
- Adapters: OTA-friendly, but cap r=4 to avoid bloat.
- Agents: Robust, but Auditor is heuristic-light for speed.

## Next: Scale to 20-qubit TT? Integrate with Or4cl3?
Fork & PR – let's riff.

Cheers,  
Architech (with a quantum coffee chaser)
