import io
import os

from setuptools import find_packages, setup

NAME = 'haxworld'
IMPORT_NAME = 'haxworld'
DESCRIPTION = ""
URL = 'https://github.com/sbl1996/haxworld'
EMAIL = 'sbl1996@gmail.com'
AUTHOR = 'Hastur'
REQUIRES_PYTHON = '>=3.10.0'
VERSION = None

REQUIRED = [
    "safetensors",
    "diffusers",
    "transformers",
    "peft",
]

here = os.path.dirname(os.path.abspath(__file__))

try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

about = {}
if not VERSION:
    with open(os.path.join(here, IMPORT_NAME, '_version.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION


setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    install_requires=REQUIRED,
    dependency_links=[],
    # include_package_data=True,
    license='MIT',
)