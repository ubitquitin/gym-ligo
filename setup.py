from setuptools import setup, find_packages

setup(
    name="gym_ligo",
    version="0.0.1",
    description="A Gymnasium wrapper for the LIGO Lightsaber physics simulation.",
    author="Rohan Khopkar",  # Update with your name
    # author_email="your.email@example.com",
    url="https://github.com/ubitquitin/gym-ligo", # Update with your repo URL

    # Automatically find the 'gym_ligo' package in this directory
    packages=find_packages(),

    # List external libraries your package needs to run
    install_requires=[
        "gymnasium>=0.29.0",
        "numpy",
        "pyyaml",
        "scipy",
        "matplotlib",
        "absl-py",
        "fancyflags",
        "pandas",
        "tqdm",
        "PyQt6"
    ],

    # Include non-python files (like your config.yaml)
    include_package_data=True,
    package_data={
        "gym_ligo": ["envs/*.yaml"],  # Ensures config.yaml is installed with the package
    },

    python_requires=">=3.8",
)