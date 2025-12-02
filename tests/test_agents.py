import pytest
import torch
import sys
sys.path.insert(0, '..')

from deploy_stub import AgentBus, ChronosDeploy
from auditor import SigmaMatrixAuditor

def test_agent_bus_registration():
    """Test agent registration and dispatch."""
    bus = AgentBus()
    
    # Register test agent
    def test_handler(msg):
        return {'echo': msg.get('data', 'none')}
    
    bus.register('test_agent', test_handler)
    
    assert 'test_agent' in bus.agents, "Agent not registered"
    print("✓ Agent registration test passed")

def test_agent_bus_dispatch():
    """Test message routing."""
    bus = AgentBus()
    
    def echo_handler(msg):
        return {'response': f"Echo: {msg.get('text', '')}"}
    
    bus.register('echo', echo_handler)
    
    response = bus.dispatch({'agent': 'echo', 'text': 'hello'})
    
    assert 'response' in response, "Response missing"
    assert 'hello' in response['response'], "Echo failed"
    print("✓ Agent dispatch test passed")

def test_agent_bus_history():
    """Test interaction history tracking."""
    bus = AgentBus()
    
    def dummy_handler(msg):
        return {'status': 'ok'}
    
    bus.register('dummy', dummy_handler)
    
    # Send multiple messages
    for i in range(5):
        bus.dispatch({'agent': 'dummy', 'id': i})
    
    history = bus.get_history(limit=3)
    
    assert len(history) == 3, "History limit not respected"
    assert history[-1]['request']['id'] == 4, "History order incorrect"
    print("✓ Agent history test passed")

def test_sigma_auditor_baseline():
    """Test auditor baseline setting."""
    auditor = SigmaMatrixAuditor(threshold=0.05)
    baseline = torch.randn(1, 128)
    
    auditor.set_baseline(baseline)
    
    assert auditor.baseline_ethic is not None, "Baseline not set"
    assert auditor.baseline_ethic.shape == baseline.shape, "Baseline shape mismatch"
    print("✓ Auditor baseline test passed")

def test_sigma_auditor_drift_measurement():
    """Test drift measurement."""
    auditor = SigmaMatrixAuditor(threshold=0.05)
    baseline = torch.randn(1, 128)
    auditor.set_baseline(baseline)
    
    # Test with identical state (zero drift)
    drift_zero = auditor.measure(baseline)
    assert drift_zero < 1e-6, "Zero drift test failed"
    
    # Test with perturbed state
    perturbed = baseline + torch.randn(1, 128) * 0.1
    drift_nonzero = auditor.measure(perturbed)
    assert drift_nonzero > 0, "Non-zero drift test failed"
    
    print(f"✓ Drift measurement test passed (drift={drift_nonzero:.6f})")

def test_sigma_auditor_dmaic_cycle():
    """Test full DMAIC cycle."""
    auditor = SigmaMatrixAuditor(threshold=0.05)
    baseline = torch.randn(1, 128)
    auditor.set_baseline(baseline)
    
    # Simulate increasing drift
    for i in range(10):
        current = baseline + torch.randn(1, 128) * (0.01 * i)
        result = auditor.dmaic_cycle(current)
        
        assert 'drift' in result, "DMAIC result missing drift"
        assert 'analysis' in result, "DMAIC result missing analysis"
        assert 'correction' in result, "DMAIC result missing correction"
    
    # Check if alert triggered
    analysis = auditor.analyze()
    print(f"✓ DMAIC cycle test passed (status={analysis['status']})")

def test_agent_handoff_latency():
    """Test agent handoff performance."""
    import time
    
    bus = AgentBus()
    
    def fast_handler(msg):
        return {'processed': True}
    
    bus.register('fast', fast_handler)
    
    # Measure handoff time
    start = time.time()
    for _ in range(100):
        bus.dispatch({'agent': 'fast', 'data': 'test'})
    elapsed = (time.time() - start) * 1000  # Convert to ms
    
    avg_latency = elapsed / 100
    assert avg_latency < 50, f"Handoff too slow: {avg_latency:.2f}ms"
    print(f"✓ Agent handoff test passed (avg={avg_latency:.2f}ms)")

def test_chronos_deploy_initialization():
    """Test deployment wrapper initialization."""
    # This test requires a model file, so we'll create a dummy
    model = torch.nn.Linear(10, 10)
    torch.jit.save(torch.jit.script(model), 'test_model.pt')
    
    try:
        deploy = ChronosDeploy('test_model.pt')
        assert deploy.model is not None, "Model not loaded"
        assert deploy.bus is not None, "Agent bus not initialized"
        print("✓ ChronosDeploy initialization test passed")
    finally:
        import os
        if os.path.exists('test_model.pt'):
            os.remove('test_model.pt')

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
