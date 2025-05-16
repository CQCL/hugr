# Configuration file for the Sphinx documentation builder.  # noqa: INP001
# See https://www.sphinx-doc.org/en/master/usage/configuration.html
import hugr

project = "HUGR Python"
copyright = "2025, Quantinuum"
author = "Quantinuum"

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

html_theme = "furo"


html_title = f"HUGR-py v{hugr.__version__} API documentation."

html_theme_options = {
    "sidebar_hide_name": False,
}

html_static_path = ["../_static"]

html_logo = "../_static/hugr_logo_no_bg.svg"

autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "conftest.py"]


intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
}

html_show_sourcelink = False
