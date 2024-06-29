from setuptools import setup, find_packages
import pathlib


PACKAGE_ROOT = pathlib.Path(__file__).parent
NAME = "Loan Approval Predictor"

VERSION_LOC = pathlib.Path.joinpath(PACKAGE_ROOT, "loan_approval_prediction", "VERSION")
with open(VERSION_LOC, 'r') as v:
    VERSION = v.read()

DESCRIPTION = "This app predicts the loan approval"

LD_LOC = pathlib.Path.joinpath(PACKAGE_ROOT, "README.md")
with open(LD_LOC, 'r') as ld:
    LONG_DESCRIPTION = ld.read()

AUTHOR = "Vasala Rahul"

AUTHOR_MAIL = "rahul.vasala98@gmail.com"

URL = "https://github.com/VRahulmech/Loan_Approval.git"

PYTHON_REQUIRES = ">=3.10.14"

REQ_LOC = pathlib.Path.joinpath(PACKAGE_ROOT, "requirements.txt")
with open(REQ_LOC, 'r') as r:
    REQ = r.read().splitlines()


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author_email=AUTHOR_MAIL,
    author=AUTHOR,
    python_requires=PYTHON_REQUIRES,
    url=URL,
    install_requires=REQ,
    packages=find_packages(exclude=('tests',)),
    package_data={"loan_approval_prediction": ['VERSION']},
    include_package_data=True,
    extras_require={},
    license="MIT",
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],

)
