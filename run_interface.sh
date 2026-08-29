#!/usr/bin/env bash
# Fusion-JEPA Studio Linux/macOS Launcher

cd "$(dirname "$0")"

echo "=============================================================================="
echo "            Fusion-JEPA Studio -- Interactive Web Interface"
echo "=============================================================================="
echo ""

# 1. Check for Virtual Environment Python
if [ -f "venv/bin/python" ]; then
    PY_EXE="venv/bin/python"
    echo "[OK] Using virtual environment Python: venv/bin/python"
elif [ -f "venv/Scripts/python.exe" ]; then
    PY_EXE="venv/Scripts/python.exe"
    echo "[OK] Using virtual environment Python: venv/Scripts/python.exe"
else
    PY_EXE="python3"
    echo "[INFO] Using system python3."
fi

# 2. Ensure Streamlit is installed
if ! "$PY_EXE" -c "import streamlit" &> /dev/null; then
    echo "[INSTALL] Streamlit not found. Installing streamlit..."
    "$PY_EXE" -m pip install streamlit
fi

echo ""
echo "=============================================================================="
echo "[LAUNCH] Starting Streamlit server..."
echo "[INFO] Opening browser at http://localhost:8501"
echo "=============================================================================="
echo ""

# 3. Launch Streamlit
"$PY_EXE" -m streamlit run app.py --server.port 8501 --server.headless false
