import sys
import os
import subprocess

req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")

with open(req_file) as file:
    for package in file:
        package = package.strip()
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
