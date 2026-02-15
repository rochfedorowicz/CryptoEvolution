# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath('..'))
# For CI: source might be in the editable install location
editable_install_path = os.path.abspath('../src/cryptoevolutionpackage')
if os.path.exists(editable_install_path):
    sys.path.insert(0, editable_install_path)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Crypto Evolution'
copyright = '2024-2026, Roch Fedorowicz'
author = 'Roch Fedorowicz'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon'
]

autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'private-members': False,
    'special-members': False,
    'inherited-members': True,
    'show-inheritance': True,
}

add_module_names = False

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_theme_options = {
    'page_width': '1400px',
    'sidebar_width': '320px',
    'body_max_width': 'none',
    'fixed_sidebar': True,
}
html_static_path = ['_static']
