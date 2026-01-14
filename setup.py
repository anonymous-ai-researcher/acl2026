"""Setup script for transformer-grokking-counting package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="transformer-grokking-counting",
    version="1.0.0",
    author="Anonymous",
    author_email="anonymous@example.com",
    description="Do Transformers Grok Succinct Algorithms? Mechanistic Evidence for Counting Circuits",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anonymous-ai-researcher/ACL2026",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "black>=23.3.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "train-transformer=scripts.train_transformer:main",
            "train-rnn=scripts.train_rnn:main",
            "analyze-model=scripts.analyze_model:main",
        ],
    },
)
