#!/bin/bash
# Architech-Chronos Setup Script

echo "========================================="
echo "Architech-Chronos Setup"
echo "========================================="

# Check Python version
python_version=$(python3.11 --version 2>&1)
echo "✓ Python: $python_version"

# Create virtual environment (optional)
read -p "Create virtual environment? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    python3.11 -m venv venv
    source venv/bin/activate
    echo "✓ Virtual environment created"
fi

# Install dependencies
echo "Installing dependencies..."
pip3 install -r requirements.txt

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Export model: python quantize_and_export.py --export ts"
echo "  2. Run tests: pytest tests/"
echo "  3. Try demo: python examples/demo_chat.py"
echo ""
