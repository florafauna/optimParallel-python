from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="optimparallel",
    version="0.0.1",
    description="A parallel version of the L-BFGS-B optimizer of scipy.optimize.minimize().",
    py_modules=["optimparallel"],
    package_dir={"": "src"},
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)",
        "Operating System :: OS Independent",
    ],
    url="https://github.com/florafauna/optimParallel-python",
    author="Florian Gerber",
    author_email="flora.fauna.gerber@gmail.com",
)
