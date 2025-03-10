#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

test_requirements = [ "tensorflow==2.3.4" , "torch==1.8.1" ]

setup(
    author="MatZaharia",
    author_email='guozh29@mail2.sysu.edu.cn',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="A scalable, portable, and lightweight Federated Learning framework.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='golf_federated',
    name='golf_federated',
    packages=find_packages(include=['golf_federated', 'golf_federated.*']),
    test_suite='example.test.test',
    tests_require=test_requirements,
    url='https://github.com/MatZaharia/generic_and_open_learning_federator',
    version='0.2.0',
    zip_safe=False,

)
