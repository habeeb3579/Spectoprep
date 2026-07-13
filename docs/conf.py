#!/usr/bin/env python
"""Sphinx configuration for SpectoPrep documentation."""

from __future__ import annotations

import os
import sys
from datetime import datetime

# Src-layout package
sys.path.insert(0, os.path.abspath("../src"))

import spectoprep  # noqa: E402

# -- Project information -----------------------------------------------------

project = "SpectoPrep"
author = "Habeeb Babatunde"
copyright = f"{datetime.now():%Y}, {author}"
version = spectoprep.__version__
release = spectoprep.__version__

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.duration",
    "sphinx.ext.autosectionlabel",
    "nbsphinx",
]

templates_path = ["_templates"]
source_suffix = {".rst": "restructuredtext"}
master_doc = "index"
language = "en"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
pygments_style = "sphinx"
pygments_dark_style = "monokai"
todo_include_todos = False

# Avoid duplicate label warnings across pages
autosectionlabel_prefix_document = True

# -- Autodoc / Napoleon ------------------------------------------------------

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "inherited-members": False,
}
# Prefer the defining module when the same symbol is re-exported at package root.
autodoc_type_aliases = {
    "PipelineOptimizer": "spectoprep.pipeline.optimizer.PipelineOptimizer",
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- nbsphinx ----------------------------------------------------------------
# The shipped tutorial uses local paths; do not execute during the docs build.
nbsphinx_execute = "never"
nbsphinx_allow_errors = True
nbsphinx_prompt_width = "0"

# -- HTML --------------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = f"{project} {release}"
html_short_title = project
html_favicon = None

html_theme_options = {
    "logo": {"text": "SpectoPrep"},
    "github_url": "https://github.com/habeeb3579/Spectoprep",
    "use_edit_page_button": False,
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "navbar_align": "left",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/spectoprep/",
            "icon": "fa-brands fa-python",
        },
        {
            "name": "GitHub",
            "url": "https://github.com/habeeb3579/Spectoprep",
            "icon": "fa-brands fa-github",
        },
    ],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
}

html_context = {
    "github_user": "habeeb3579",
    "github_repo": "Spectoprep",
    "github_version": "main",
    "doc_path": "docs",
}

htmlhelp_basename = "spectoprepdoc"

# -- LaTeX / man / texinfo (kept for completeness) ---------------------------

latex_documents = [
    (master_doc, "spectoprep.tex", "SpectoPrep Documentation", author, "manual"),
]
man_pages = [(master_doc, "spectoprep", "SpectoPrep Documentation", [author], 1)]
texinfo_documents = [
    (
        master_doc,
        "spectoprep",
        "SpectoPrep Documentation",
        author,
        "spectoprep",
        "Bayesian optimization of spectroscopic preprocessing pipelines.",
        "Miscellaneous",
    ),
]
