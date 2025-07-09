"""
Script to run the ChanceRAG Streamlit application
"""
import subprocess
import sys

def run_streamlit_app():
    """Run the Streamlit application"""
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit app: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nStreamlit app stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    run_streamlit_app()
