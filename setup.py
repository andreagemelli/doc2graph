from setuptools import find_packages, setup
from setuptools_git_versioning import version_from_git
import os

HERE = os.path.dirname(os.path.abspath(__file__))

def parse_requirements(file_content):
    lines = file_content.splitlines()
    return [line.strip() for line in lines if line and not line.startswith("#")]

with open(os.path.join(HERE, "requirements.txt")) as f:
    requirements = parse_requirements(f.read())

with open("src/root.env", "w") as f:
    f.write(f"ROOT = '{HERE}'")

setup(
    name='doc2graph',
    version=version_from_git(),
    packages=['doc2graph'],
    package_dir={'doc2graph': 'src'},
    description='Repo to transform Documents to Graphs, performing several tasks on them',
    author='Andrea Gemelli',
    license='MIT',
    keywords="document analysis graph",
    setuptools_git_versioning={
        "enabled": True,
    },
    python_requires=">=3.7",
    setup_requires=["setuptools_git_versioning"],
    install_requires=requirements,
)
