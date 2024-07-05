# Configuration file for the Sphinx documentation builder.  # noqa: INP001
# See https://www.sphinx-doc.org/en/master/usage/configuration.html


project = "HUGR Python"
copyright = "2024, Quantinuum"
author = "Quantinuum"

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_multiversion",
]

html_theme = "furo"
html_title = "HUGR python package API documentation."
html_theme_options = {}


templates_path = ["_templates", "../quantinuum-sphinx/_templates"]
html_static_path = ['../_static', '../quantinuum-sphinx/_static']

autosummary_generate = True

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "conftest.py"]

smv_branch_whitelist = "main"
smv_tag_whitelist = r"^hugr-py-.*$"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
}

html_show_sourcelink = False
