from setuptools import setup, find_packages

setup(
    name="ai-meeting-assistant",
    version="0.1.0",
    packages=find_packages(
        exclude=["tests*", "models*", ".venv*"]
    ),
)
