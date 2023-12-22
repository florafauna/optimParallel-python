from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="optimparallel",
    version="0.1.3",
    description="A parallel version of the L-BFGS-B optimizer of scipy.optimize.minimize().",
    py_modules=["optimparallel"],
    package_dir={"": "src"},
    install_requires=["scipy", "numpy"],
    extras_require={"dev": ["pytest"]},
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    url="https://github.com/florafauna/optimParallel-python",
    author="Florian Gerber",
    author_email="flora.fauna.gerber@gmail.com",
)
