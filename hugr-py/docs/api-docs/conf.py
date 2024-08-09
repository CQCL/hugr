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
]

html_theme = "sphinx_book_theme"

html_title = "HUGR python package API documentation."

html_theme_options = {
    "repository_url": "https://github.com/CQCL/hugr",
    "use_repository_button": True,
    "navigation_with_keys": True,
    "logo": {
        "image_light": "_static/Quantinuum_logo_black.png",
        "image_dark": "_static/Quantinuum_logo_white.png",
    },
    "show_navbar_depth": 2,
}

html_static_path = ["../_static"]
html_css_files = ["custom.css"]

autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "conftest.py"]


intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
}

html_show_sourcelink = False
