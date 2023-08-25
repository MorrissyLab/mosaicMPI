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
import os
import sys
from importlib import metadata
sys.path.insert(0, os.path.abspath('../../src/mosaicmpi'))


# -- Project information -----------------------------------------------------

project = 'mosaicMPI'
copyright = '2023, Theodore Verhey'
author = 'Theodore Verhey'

# The full version, including alpha/beta/rc tags
release = '1.9.0'
release = version = metadata.version("mosaicmpi")


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx_copybutton"
    ]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output ----------------------------------------------

html_static_path = ["_static"]
html_theme = "sphinx_book_theme"
# html_theme = 'sphinx_rtd_theme'
html_theme_options = dict(
    use_repository_button=True,
    repository_url="https://github.com/MorrissyLab/mosaicMPI",
    repository_branch="main",
)
html_logo = "_static/img/logo_only.svg"
issues_github_path = "MorrissyLab/mosaicMPI"
html_show_sphinx = False


# autosummary_generate = True
autoclass_content = 'both'
autodoc_member_order = 'bysource'
