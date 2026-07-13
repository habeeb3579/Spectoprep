.. highlight:: shell

============
Installation
============

Requirements
------------

* Python **3.10** or newer
* Scientific stack: NumPy, SciPy, scikit-learn, pandas, Matplotlib, seaborn
* Bayesian optimization, Typer, Rich, and structlog (installed automatically)

Stable release
--------------

Install from PyPI:

.. code-block:: console

   $ pip install spectoprep

Or from the Anaconda.org channel ``habeebest`` (not conda-forge):

.. code-block:: console

   $ conda install -c habeebest spectoprep

Tagged releases publish to both PyPI and the ``habeebest`` conda channel.

Verify the install:

.. code-block:: console

   $ spectoprep version
   $ python -c "import spectoprep; print(spectoprep.__version__)"

From source
-----------

Clone the repository and install in editable mode with development extras:

.. code-block:: console

   $ git clone https://github.com/habeeb3579/Spectoprep.git
   $ cd Spectoprep
   $ python -m pip install -e ".[dev]"

For documentation builds only:

.. code-block:: console

   $ python -m pip install -e ".[docs]"
   $ cd docs && make html

Building the docs
-----------------

.. code-block:: console

   $ cd docs
   $ make html
   $ python -m http.server --directory _build/html

Open ``http://localhost:8000`` in a browser. Pandoc is required for notebooks
(``nbsphinx``); on macOS: ``brew install pandoc``.
