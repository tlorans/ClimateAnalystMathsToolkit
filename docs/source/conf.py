# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'Climate Analyst Maths Toolkit'
copyright = '2023, Thomas Lorans'
author = 'Thomas Lorans'

# The full version, including alpha/beta/rc tags
release = '0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'myst_parser',
    'sphinx.ext.autosectionlabel',
    "nbsphinx",
    "sphinxcontrib.bibtex",
]

bibtex_bibfiles = ['refs.bib']
myst_enable_extensions = ["dollarmath", "amsmath"]

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_book_theme'
html_static_path = []