import setuptools
from pathlib import Path
from setuptools.command.build_py import build_py


CROSS_SECTION_FILES = (
    "Arplus_Ar_elastic_isotropic_scattering.csv",
    "Arplus_Ar_resonant_charge_exchange.csv",
)


class BuildPyWithCrossSections(build_py):
    """Place the repository data tables inside the built Python package."""

    def run(self):
        super().run()
        source = Path(__file__).parent / "data" / "cross_sections"
        destination = Path(self.build_lib) / "opendust_jax" / "data" / "cross_sections"
        destination.mkdir(parents=True, exist_ok=True)
        for filename in CROSS_SECTION_FILES:
            self.copy_file(str(source / filename), str(destination / filename))

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
    package_data={"opendust_jax": ["cuda/*.cu", "_cuda/*.so"]},
    cmdclass={"build_py": BuildPyWithCrossSections},
    version = "1.0.0",
    
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
