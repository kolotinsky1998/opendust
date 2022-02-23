import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "opendust",
    description = "OpenDust: A fast GPU-accelerated code for calculation forces, acting on micro-particles in a plasma flow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url = "https://github.com/kolotinsky1998/opendust",
    author = "Daniil Kolotinskii",
    author_email = "kolotinskiy.da@phystech.edu",
    license = "MIT",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),    
    version = "1.0.0",
    
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
