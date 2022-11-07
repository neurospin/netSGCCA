#! /usr/bin/env python
##########################################################################
# Copyright (C) CEA, 2022
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# coxgnet current version
version_major = 0
version_minor = 1
version_micro = 0

# Expected by setup.py: string of form "X.Y.Z"
__version__ = "{0}.{1}.{2}".format(version_major, version_minor, version_micro)


# Define default coxgnet path for the package


# Expected by setup.py: the status of the project
CLASSIFIERS = ["Development Status :: 5 - Production/Stable",
               "Environment :: Console",
               "Environment :: X11 Applications :: Qt",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering",
               "Topic :: Utilities"]

# Project descriptions
description = """
API for NetSGCCA.
"""
SUMMARY = """
Offers API for NetSGCCA approach based on pylearn-parsimony.
"""
long_description = """
============
coxgnet
============

Offers API for COX SGCCA approach based on pylearn-parsimony.
"""

# Main setup parameters
NAME = "NetSGCCA"
ORGANISATION = "CEA"
MAINTAINER = "Vincent Frouin"
MAINTAINER_EMAIL = "vincent.frouin@cea.fr"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "https://gitlab.com/brainomics/coxgnet"
DOWNLOAD_URL = "https://gitlab.com/brainomics/coxgnet"
LICENSE = "CeCILL-B"
CLASSIFIERS = CLASSIFIERS
AUTHOR = "coxgnet developers"
AUTHOR_EMAIL = "vincent.frouin@cea.fr"
PLATFORMS = "OS Independent"
ISRELEASE = True
VERSION = __version__
PROVIDES = ["coxgnet"]
REQUIRES = [
    "pandas>=0.23.4",
    "packaging"
]
EXTRA_REQUIRES = {}
SCRIPTS = [
    
]
