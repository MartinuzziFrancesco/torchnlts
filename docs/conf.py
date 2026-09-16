# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup -----------------------------------------------------

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "torchnlts"
html_title = "torchnlts"
copyright = "2026, Francesco Martinuzzi"
author = "Francesco Martinuzzi"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False
autodoc_typehints = "none"
autodoc_typehints_format = "short"
autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "use_edit_page_button": True,
    "show_nav_level": 2,
    "github_url": "https://github.com/MartinuzziFrancesco/torchnlts",
    "header_links_before_dropdown": 10,
    "external_links": [{"name": "GitHub", "url": "https://github.com/MartinuzziFrancesco/torchnlts"}],
}

html_show_sourcelink = False
html_context = {
    "github_user": "MartinuzziFrancesco",
    "github_repo": "torchnlts",
    "github_version": "main",
    "doc_path": "docs",
}
