from setuptools import find_packages, setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='Medical Chatbot',
    version='0.0.0',
    author='smriti',
    author_email='smriti003.sharma@gamil.com',
    packages=find_packages(),
    install_requires=requirements
)
