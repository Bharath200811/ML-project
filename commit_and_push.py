"""
Git Commit and Push Helper for Refactored Forecasting Modules
"""

import os
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_URL = "https://github.com/Bharath200811/ML-project.git"

# Remove scratch execute script before commit
if os.path.exists(os.path.join(BASE_DIR, "execute.py")):
    os.remove(os.path.join(BASE_DIR, "execute.py"))

def run_cmd(cmd):
    print(f"\n> Running command: {cmd}")
    res = subprocess.run(
        cmd,
        cwd=BASE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True
    )
    if res.stdout:
        print(res.stdout.strip())
    if res.stderr:
        print(res.stderr.strip())
    return res.returncode

if __name__ == "__main__":
    print("=== COMMITTING AND PUSHING REFACTORED MODULES ===")
    
    run_cmd("git add .")
    run_cmd('git commit -m "Separate training and testing modules"')
    code = run_cmd("git push origin main")
    
    if code == 0:
        print("\n=========================================================")
        print("SUCCESS: Refactored modules committed & pushed to GitHub!")
        print("Repository URL: https://github.com/Bharath200811/ML-project")
        print("=========================================================")
    else:
        print("\nPush output printed above.")
