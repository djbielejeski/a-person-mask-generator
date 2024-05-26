import sys
import os
import subprocess

def install_package(package_name):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])


req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")

with open(req_file) as file:
    for package in file:
        package = package.strip()
        try:
            __import__(package)
        except ImportError:
            install_package(package)

from .a_person_mask_generator_comfyui import APersonMaskGenerator

NODE_CLASS_MAPPINGS = {
    "APersonMaskGenerator": APersonMaskGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "APersonMaskGenerator": "A Person Mask Generator"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']