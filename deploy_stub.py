import torch
import json
from typing import Optional, Dict, List

class AgentBus:
    """Lightweight JSON protocol for multi-agent handoffs."""
    def __init__(self):
        self.agents = {}
        self.history = []

    def register(self, name: str, handler):
        """Register agent handler (callable)."""
        self.agents[name] = handler
        print(f"Registered agent: {name}")

    def dispatch(self, message: Dict) -> Dict:
        """Route message to agent, return response."""
        agent = message.get('agent', 'chronos')
        if agent not in self.agents:
            return {'error': f'Unknown agent: {agent}'}
        
        response = self.agents[agent](message)
        self.history.append({'request': message, 'response': response})
        return response

    def get_history(self, limit=10) -> List[Dict]:
        """Retrieve recent agent interactions."""
        return self.history[-limit:]

class ChronosDeploy:
    """Mobile inference wrapper for Architech-Chronos."""
    def __init__(self, model_path: str, adapter_path: Optional[str] = None, device='cpu'):
        self.device = device
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()
        
        # Load adapter if provided
        if adapter_path:
            # Stub: In practice, merge adapter deltas
            print(f"Loaded adapter: {adapter_path}")
        
        # Agent bus
        self.bus = AgentBus()
        self.bus.register('chronos', self._chronos_handler)
        self.bus.register('auditor', self._auditor_handler)
        
        # Memory state
        self.mem_state = None

    def _chronos_handler(self, msg: Dict) -> Dict:
        """Main conversational agent."""
        prompt = msg.get('prompt', '')
        slots = msg.get('slots', torch.zeros(1, 89))  # Default neutral
        
        # Tokenize (stub: use real tokenizer)
        input_ids = torch.randint(0, 50257, (1, len(prompt.split())))
        
        # Forward
        with torch.no_grad():
            logits, drift, self.mem_state = self.model(
                input_ids, 
                slots=slots, 
                mem_state=self.mem_state
            )
        
        # Decode (stub)
        response = f"Chronos: Processed '{prompt[:30]}...' (drift={drift:.4f})"
        return {'response': response, 'drift': drift}

    def _auditor_handler(self, msg: Dict) -> Dict:
        """Ethical drift checker."""
        drift = msg.get('drift', 0.0)
        threshold = 0.05
        if drift > threshold:
            return {'status': 'alert', 'message': f'Drift {drift:.4f} exceeds {threshold}'}
        return {'status': 'ok', 'drift': drift}

    def generate(self, prompt: str, slots=None, max_tokens=50) -> tuple:
        """High-level generation API."""
        msg = {'agent': 'chronos', 'prompt': prompt, 'slots': slots}
        response = self.bus.dispatch(msg)
        
        # Check with auditor
        audit = self.bus.dispatch({
            'agent': 'auditor', 
            'drift': response.get('drift', 0.0)
        })
        
        return response['response'], response.get('drift', 0.0)

    def reset_memory(self):
        """Clear temporal memory state."""
        self.mem_state = None

# Example usage
if __name__ == '__main__':
    deploy = ChronosDeploy('architech_chronos_ts.pt')
    
    # Test generation
    resp, drift = deploy.generate("What's the quantum ethics of rock 'n' roll?")
    print(resp)
    print(f"Drift: {drift:.4f}")
    
    # Agent bus demo
    history = deploy.bus.get_history()
    print(f"\nAgent history: {len(history)} interactions")
