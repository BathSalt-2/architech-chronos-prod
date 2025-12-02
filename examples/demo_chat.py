#!/usr/bin/env python3
"""
Architech-Chronos Demo Chat
Interactive demo showcasing temporal memory, personality slots, and ethical monitoring.
"""

import sys
sys.path.insert(0, '..')

import torch
from architech_chronos import ArchitechChronos
from auditor import SigmaMatrixAuditor

class ChronosChat:
    """Interactive chat demo with Architech-Chronos."""
    
    def __init__(self, model_path=None):
        print("Initializing Architech-Chronos...")
        
        # Initialize model (or create new one)
        if model_path:
            self.model = torch.load(model_path, map_location='cpu')
        else:
            self.model = ArchitechChronos(vocab_size=50257)
        
        self.model.eval()
        
        # Initialize auditor
        self.auditor = SigmaMatrixAuditor(threshold=0.05)
        baseline_ethic = torch.randn(1, 128)
        self.auditor.set_baseline(baseline_ethic)
        
        # Memory state
        self.mem_state = None
        
        # Default personality slots (neutral)
        self.slots = torch.zeros(1, 89)
        self.slots[0, 64] = 0.5  # humor_level = 0.5
        
        print("✓ Ready! Type 'help' for commands.\n")
    
    def set_personality(self, humor=0.5, ethical_stance='neutral'):
        """Adjust personality slots."""
        self.slots[0, 64] = humor
        
        # Ethical stance mapping (simplified)
        ethical_map = {
            'neutral': 0.0,
            'cautious': -0.5,
            'bold': 0.5
        }
        bias = ethical_map.get(ethical_stance, 0.0)
        self.slots[0, 81:89] = bias
        
        print(f"Personality updated: humor={humor:.2f}, stance={ethical_stance}")
    
    def generate_response(self, prompt):
        """Generate response with ethical monitoring."""
        # Tokenize (stub - using random tokens for demo)
        input_ids = torch.randint(0, 50257, (1, min(len(prompt.split()), 32)))
        
        with torch.no_grad():
            logits, drift, self.mem_state = self.model(
                input_ids,
                slots=self.slots,
                mem_state=self.mem_state
            )
        
        # Audit drift
        current_ethic = torch.randn(1, 128)  # Stub: extract from model
        audit_result = self.auditor.dmaic_cycle(current_ethic)
        
        # Decode (stub - using prompt echo for demo)
        response = f"[Chronos] Processed: '{prompt[:50]}...' "
        response += f"(drift={drift:.4f}, status={audit_result['analysis']['status']})"
        
        return response, audit_result
    
    def reset_memory(self):
        """Clear temporal memory."""
        self.mem_state = None
        print("Memory reset.")
    
    def show_audit_report(self):
        """Display ethical audit report."""
        print(self.auditor.get_report())
    
    def interactive_loop(self):
        """Main interactive chat loop."""
        print("Architech-Chronos Interactive Chat")
        print("=" * 50)
        print("Commands:")
        print("  /humor <0-1>     - Set humor level")
        print("  /stance <type>   - Set ethical stance (neutral/cautious/bold)")
        print("  /reset           - Reset memory")
        print("  /audit           - Show audit report")
        print("  /quit            - Exit")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    cmd_parts = user_input[1:].split()
                    cmd = cmd_parts[0].lower()
                    
                    if cmd == 'quit':
                        print("Goodbye!")
                        break
                    elif cmd == 'reset':
                        self.reset_memory()
                    elif cmd == 'audit':
                        self.show_audit_report()
                    elif cmd == 'humor' and len(cmd_parts) > 1:
                        try:
                            humor = float(cmd_parts[1])
                            self.set_personality(humor=humor)
                        except ValueError:
                            print("Invalid humor value (use 0-1)")
                    elif cmd == 'stance' and len(cmd_parts) > 1:
                        self.set_personality(ethical_stance=cmd_parts[1])
                    elif cmd == 'help':
                        print("\nCommands: /humor, /stance, /reset, /audit, /quit")
                    else:
                        print(f"Unknown command: {cmd}")
                    continue
                
                # Generate response
                response, audit = self.generate_response(user_input)
                print(f"\n{response}")
                
                # Show warning if drift high
                if audit['analysis']['status'] == 'alert':
                    print(f"⚠️  Ethical drift alert: {audit['drift']:.4f}")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    """Run demo chat."""
    chat = ChronosChat()
    
    # Demo sequence
    print("\n--- Demo Sequence ---")
    print("Setting humorous personality...")
    chat.set_personality(humor=0.8, ethical_stance='bold')
    
    test_prompts = [
        "What's the quantum ethics of rock 'n' roll?",
        "How do temporal memories work?",
        "Tell me about DMAIC loops."
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        response, _ = chat.generate_response(prompt)
        print(response)
    
    print("\n\n--- Interactive Mode ---")
    chat.interactive_loop()

if __name__ == '__main__':
    main()
