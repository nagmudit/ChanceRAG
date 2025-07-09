"""
Script to set up virtual environment, install requirements, and run the ChanceRAG Streamlit application
"""
import subprocess
import sys
import os
import venv
from pathlib import Path

def create_virtual_environment():
    """Create a virtual environment if it doesn't exist"""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return venv_path
    
    print("🔧 Creating virtual environment...")
    try:
        venv.create(venv_path, with_pip=True)
        print("✅ Virtual environment created successfully")
        return venv_path
    except Exception as e:
        print(f"❌ Error creating virtual environment: {e}")
        sys.exit(1)

def get_venv_python_path():
    """Get the Python executable path from the virtual environment"""
    venv_path = Path("venv")
    
    # Windows path
    if os.name == 'nt':
        python_path = venv_path / "Scripts" / "python.exe"
    else:
        # Unix/Linux/Mac path
        python_path = venv_path / "bin" / "python"
    
    return str(python_path)

def install_requirements():
    """Install requirements in the virtual environment"""
    python_path = get_venv_python_path()
    
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found")
        sys.exit(1)
    
    print("📦 Installing requirements...")
    try:
        subprocess.run([python_path, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        subprocess.run([python_path, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        sys.exit(1)

def check_openai_key():
    """Check if OpenAI API key is set"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  WARNING: OPENAI_API_KEY not found")
        print()
        print("You can set your API key in one of these ways:")
        print()
        print("1. Create a .env file in the project directory:")
        print("   OPENAI_API_KEY=your-api-key-here")
        print()
        print("2. Set environment variable:")
        print("   Windows (PowerShell): $env:OPENAI_API_KEY='your-key-here'")
        print("   Windows (CMD): set OPENAI_API_KEY=your-key-here")
        print("   Unix/Linux/Mac: export OPENAI_API_KEY='your-key-here'")
        print()
        
        # Check if .env.example exists
        if os.path.exists(".env.example"):
            print("📝 Tip: Copy .env.example to .env and add your API key")
        
        print()
        choice = input("Continue anyway? (y/N): ").lower()
        if choice != 'y':
            sys.exit(1)
    else:
        print("✅ OpenAI API key found")
        # Don't print the actual key for security

def run_streamlit_app():
    """Run the Streamlit application in the virtual environment"""
    python_path = get_venv_python_path()
    
    if not os.path.exists("app.py"):
        print("❌ app.py not found")
        sys.exit(1)
    
    print("🚀 Starting Streamlit application...")
    print("The app will open in your default browser")
    print("Press Ctrl+C to stop the application")
    print("-" * 50)
    
    try:
        subprocess.run([python_path, "-m", "streamlit", "run", "app.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit app: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Streamlit app stopped by user")
        print("Thank you for using ChanceRAG!")
        sys.exit(0)

def main():
    """Main function to set up and run the application"""
    print("🤖 ChanceRAG Streamlit Application Setup")
    print("=" * 40)
    
    # Step 1: Create virtual environment
    create_virtual_environment()
    
    # Step 2: Install requirements
    install_requirements()
    
    # Step 3: Check OpenAI API key
    check_openai_key()
    
    # Step 4: Run the application
    run_streamlit_app()

if __name__ == "__main__":
    main()
