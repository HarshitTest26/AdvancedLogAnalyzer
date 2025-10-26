#!/usr/bin/env python3
"""
Startup script for Advanced Log Reader
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # Get the directory of this script
    script_dir = Path(__file__).parent
    
    # Change to the script directory
    os.chdir(script_dir)
    
    # Set up the Python path
    venv_python = script_dir / ".venv" / "bin" / "python"
    
    if not venv_python.exists():
        print("❌ Virtual environment not found. Please run:")
        print("   python -m venv .venv")
        print("   source .venv/bin/activate")
        print("   pip install -r requirements.txt")
        return 1
    
    print("🚀 Starting Advanced Log Reader...")
    print("📁 Working directory:", script_dir)
    print("🐍 Using Python:", venv_python)
    print()
    print("Once the server starts:")
    print("📖 Open your browser to http://localhost:8501")
    print("⌨️  Press Ctrl+C to stop the server")
    print()
    
    try:
        # Run the Streamlit app
        cmd = [str(venv_python), "-m", "streamlit", "run", "app.py", "--server.port=8501"]
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down Advanced Log Reader. Goodbye!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting the application: {e}")
        return 1
    except FileNotFoundError:
        print("❌ Streamlit not found. Please install requirements:")
        print("   pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())