import os


def install_package(package_name):
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])


def install_dependencies():
    req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")

    with open(req_file) as file:
        for package in file:
            package = package.strip()
            try:
                __import__(package)
            except ImportError:
                install_package(f"{package}")


install_dependencies()
