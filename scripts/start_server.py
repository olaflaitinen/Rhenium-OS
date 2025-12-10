import uvicorn
import os
import sys

# Add repo root to path so modules are found
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def start():
    print(">>> Starting Rhenium OS Neural Node...")
    # Run the server
    # 'rhenium.server.app:app' refers to the FastAPI instance in rhenium/server/app.py
    uvicorn.run("rhenium.server.app:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    start()
