from setuptools import setup

setup(
    name="XTEC-GPU",
    version="1.0.0",
    description="X-ray Temperature Clustering — GPU-accelerated",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/KimGroup/XTEC-GPU",
    author="Jordan Venderley",
    maintainer="Krishnanand Mallayya",
    maintainer_email="krishnanandmallayya@gmail.com",
    license="MIT",
    packages=["xtec_gpu", "xtec_gpu.plugins"],
    package_dir={
        "xtec_gpu": "src",
        "xtec_gpu.plugins": "src/plugins",
    },
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "torch",
        "torchgmm",
        "nexusformat",
        "h5py",
    ],
    extras_require={
        "nexpy": ["nexpy"],
    },
    entry_points={
        "console_scripts": [
            "xtec-gpu=xtec_gpu.xtec_cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
