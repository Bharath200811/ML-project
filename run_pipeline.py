"""
Pipeline Execution Runner & Logger
"""

import sys
import io
import os

# Redirect output to both console and log file
log_file_path = os.path.join("outputs", "pipeline_execution.log")
os.makedirs("outputs", exist_ok=True)

class TeeStream:
    def __init__(self, stream1, stream2):
        self.stream1 = stream1
        self.stream2 = stream2

    def write(self, data):
        self.stream1.write(data)
        self.stream2.write(data)

    def flush(self):
        self.stream1.flush()
        self.stream2.flush()

if __name__ == "__main__":
    with open(log_file_path, "w", encoding="utf-8") as f:
        sys.stdout = TeeStream(sys.__stdout__, f)
        sys.stderr = TeeStream(sys.__stderr__, f)
        
        # Import and execute main pipeline
        import main
        main.main()
