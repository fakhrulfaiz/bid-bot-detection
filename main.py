import subprocess
import sys

if __name__ == "__main__":
    result = subprocess.run(["dvc", "repro"], check=False)
    sys.exit(result.returncode)
