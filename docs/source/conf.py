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
import importlib
import inspect
import sys
from datetime import date
from pathlib import Path
import mock   
MOCK_MODULES = ['numpy', 'scipy', 'matplotlib', 'matplotlib.pyplot', 'scipy.interpolate','torch','pandas','scikit-learn'] 
for mod_name in MOCK_MODULES: 
    sys.modules[mod_name] = mock.Mock()
# import git

sys.path.insert(0, str(Path("../..").absolute()))
sys.path.insert(0, os.path.abspath('../../code'))
sys.path.append(os.path.abspath('../../Tutorial'))
# # from CANAL import __version__
# sys.path.insert(0, os.path.abspath('../../Tutorial'))

# -- Project information -----------------------------------------------------

project = 'CANAL'
copyright = '2023, aster-ww'
author = 'aster-ww'

# The full version, including alpha/beta/rc tags
release = 'v1.0.0'
extensions = [     "sphinx_rtd_theme",     "nbsphinx",     "sphinx.ext.autodoc",     "sphinx.ext.napoleon",        "sphinx_copybutton",     "myst_parser", ]
source_suffix = {'.rst': 'restructuredtext', '.txt': 'markdown', '.md': 'markdown', }
myst_enable_extensions = ["tasklist", "deflist", "dollarmath", ]
templates_path = ['_templates']
exclude_patterns = []
language = 'en'
html_theme = 'sphinx_rtd_theme'
# html_theme_options = {'analytics_anonymize_ip': False, 'logo_only': True, 'display_version': True,
#                       'prev_next_buttons_location': 'bottom', 'style_external_links': False,
#                       'collapse_navigation': True, 'sticky_navigation': True, 'navigation_depth': 4,
#                       'includehidden': True, 'titles_only': False, }
html_theme_options = {
    # "analytics_id": "G-XXXXXXXXXX",  # Provided by Google in your dashboard
    "analytics_anonymize_ip": False,
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "vcs_pageview_mode": "",
    # "style_nav_header_background": "white",
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

html_logo = "./_static/logo.png"
html_static_path = ['_static']
html_js_files = ['my_custom.js', ]
autoapi_dirs = ["../../code"]

def setup(app):
    app.add_css_file('my_theme.css')
