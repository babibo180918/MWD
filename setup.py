import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = [
        "scikit-image",
        "pydot",
        "scikit-learn",
        "scipy",
        "opencv-python>=2.4.9",
        "matplotlib>=3.2.1",
        "setuptools>=47.1.1",
        "pyyaml>=5.3.1"
]

setuptools.setup(
        name="MWD",
        version="1.0.0",
        author="Nhan Nguyen",
        author_email="nguyendtnhan@gmail.com",
        install_requires=requirements,
        description="practice project used to detect mask wearing face"
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/babibo180918/MWD"
        packages=setuptools.find_packages(include=["MWD*"]),
        classifiers=[
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Intended Audience :: Science/Research",
            "Operating System :: POSIX :: Linux",
            "License :: OSI Approved :: Apache Software License",
            "Topic :: Software Development :: Libraries :: Python Modules"
        ],
        python_requires='>=3.6'
)

