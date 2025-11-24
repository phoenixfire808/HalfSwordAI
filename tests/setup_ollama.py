"""
Quick setup script to ensure Ollama and Qwen model are ready
"""
import requests
import subprocess
import sys

def check_ollama_running():
    """Check if Ollama service is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def check_qwen_model():
    """Check if Qwen model is available"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            qwen_models = [m for m in models if "qwen" in m.get("name", "").lower()]
            return len(qwen_models) > 0
    except:
        pass
    return False

def pull_qwen_model():
    """Pull Qwen model if not available"""
    print("Pulling Qwen model...")
    try:
        subprocess.run(["ollama", "pull", "qwen2.5:latest"], check=True)
        print("Qwen model pulled successfully!")
        return True
    except subprocess.CalledProcessError:
        print("Failed to pull model. Please run manually: ollama pull qwen2.5:latest")
        return False
    except FileNotFoundError:
        print("Ollama not found. Please install Ollama from https://ollama.com")
        return False

def main():
    print("Checking Ollama setup...")
    
    if not check_ollama_running():
        print("❌ Ollama service is not running!")
        print("Please start Ollama: ollama serve")
        sys.exit(1)
    
    print("✅ Ollama service is running")
    
    if not check_qwen_model():
        print("⚠️  Qwen model not found")
        response = input("Pull Qwen model now? (y/n): ")
        if response.lower() == 'y':
            if not pull_qwen_model():
                sys.exit(1)
        else:
            print("Please run: ollama pull qwen2.5:latest")
            sys.exit(1)
    else:
        print("✅ Qwen model is available")
    
    print("\n✅ Setup complete! You can now run: python main.py")

if __name__ == "__main__":
    main()

