"""
setup.py for LLMOptimize v3.4.0
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llmoptimize",
    version="3.4.0",
    author="LLMOptimize Team",
    author_email="hackrudra@gmail.com",
    description="Reduce LLM costs by 90% - AI recommendations with NO API keys needed!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hackrudra1234/llmoptimize",

    # FIX 1: auto-discover llmoptimize + any subpackages
    packages=find_packages(include=["llmoptimize", "llmoptimize.*"]),

    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Monitoring",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
        "Environment :: Console",
    ],
    python_requires=">=3.8",

    # FIX 2: removed 'requests' — SDK uses only stdlib urllib (no extra deps needed)
    install_requires=[],

    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
        ],
        "server": [
            "fastapi>=0.104.1",
            "uvicorn[standard]>=0.24.0",
            "sqlalchemy>=2.0.23",
            "psycopg2-binary>=2.9.9",
            "pydantic>=2.0.0",
        ],
        "full": [
            "anthropic>=0.3.0",
            "groq>=0.4.0",
            "langchain>=0.1.0",
        ],
    },

    # FIX 3: CLI entry point — enables 'llmoptimize audit mycode.py' after pip install
    entry_points={
        "console_scripts": [
            "llmoptimize=llmoptimize.cli:main",
        ],
    },

    keywords=[
        "ai", "llm", "cost", "optimization", "tracking",
        "openai", "anthropic", "claude", "gpt", "groq",
        "ml", "recommendations", "no-api-key",
        "cost-reduction", "savings",
    ],
    project_urls={
        "Homepage": "https://aioptimize.up.railway.app",
        "Source": "https://github.com/hackrudra1234/llmoptimize",
        "Bug Reports": "https://github.com/hackrudra1234/llmoptimize/issues",
    },
    license="Proprietary",
    include_package_data=True,
    zip_safe=False,
)