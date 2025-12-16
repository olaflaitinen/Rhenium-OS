
import os
import subprocess
import time
import sys
import threading

def run_server():
    print("Starting API Server...")
    # Using uvicorn directly
    cmd = [sys.executable, "-m", "uvicorn", "rhenium.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + "src"
    
    proc = subprocess.Popen(cmd, env=env)
    return proc

def run_java_client():
    print("Compiling Java Client...")
    if not os.path.exists("java/RheniumClient.java"):
        print("Error: Java client not found")
        return
        
    subprocess.run(["javac", "java/RheniumClient.java"], check=True)
    
    print("Running Java Client...")
    subprocess.run(["java", "-cp", "java", "RheniumClient"], check=True)

def main():
    server_proc = run_server()
    
    try:
        # Wait for server to start
        print("Waiting for server to initialize (10s)...")
        time.sleep(10)
        
        # Run Java client
        run_java_client()
        
    finally:
        print("Stopping Server...")
        server_proc.terminate()
        server_proc.wait()

if __name__ == "__main__":
    main()
