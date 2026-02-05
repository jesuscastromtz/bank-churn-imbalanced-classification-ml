#!/usr/bin/env bash
# Setup & run script for Bank Churn Prediction Project
# Time: ~5 minutes total

echo "🏦 Bank Churn Prediction - Setup Script"
echo "========================================"
echo ""

# Step 1: Install dependencies
echo "1️⃣  Installing dependencies..."
pip install -r requirements.txt -q
echo "   ✓ Dependencies installed"
echo ""

# Step 2: Run basic tests
echo "2️⃣  Running basic tests..."
python test_basic.py
if [ $? -eq 0 ]; then
    echo "   ✓ All tests passed"
else
    echo "   ❌ Tests failed"
    exit 1
fi
echo ""

# Step 3: Launch notebook
echo "3️⃣  Launching Jupyter notebook..."
echo "   Opening: notebooks/ (3 narrative notebooks)"
echo ""
jupyter notebook notebooks/

echo ""
echo "✅ Setup complete!"
echo ""
echo "📖 Next steps:"
echo "   1. Review README.md for project overview"
echo "   2. Execute notebooks in order: 1_problema → 2_solucion → 3_resultados"
echo "   3. Results will appear in ~3 minutes"
echo ""
