import launch
import os

req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")

with open(req_file) as file:
    for package in file:
        try:
            package = package.strip()
            if not launch.is_installed(package):
                launch.run_pip(f"install {package}", f"a-person-mask-generator requirement: {package}")
        except Exception as e:
            print(e)
            print(f'a-person-mask-generator: Warning: Failed to install {package}.')