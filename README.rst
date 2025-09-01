HTTomolib is a library of methods for tomography
-------------------------------------------------

**HTTomolib** is a collection of CPU-only image processing methods in Python for computed tomography.

**HTTomolib** can be used as a stand-alone library, however, it has been specifically developed to 
work together with the `HTTomo <https://diamondlightsource.github.io/httomo/>`_ package.
HTTomo is a user interface (UI) written in Python for fast big data processing using MPI protocols.
**HTTomolib** methods for processing using GPU are accessible in the dedicated
`HTTomolibGPU <https://github.com/DiamondLightSource/httomolibgpu>`_ repository. 

Purpose of HTTomolib
====================

**HTTomolib** can be used as a stand-alone library, but it has been specifically developed to 
work together with the `HTTomo <https://diamondlightsource.github.io/httomo/>`_ package.
HTTomo is a user interface (UI) written in Python for fast big data processing using MPI protocols. 

Installation
============

HTTomolib is available on PyPI, so it can be installed into either a virtual environment or a
conda environment.

Virtual environment
~~~~~~~~~~~~~~~~~~~
.. code-block:: console

   $ python -m venv httomolib
   $ source httomolib/bin/activate
   $ pip install httomolib

Conda environment
~~~~~~~~~~~~~~~~~
.. code-block:: console

   $ conda create --name httomolib # create a fresh conda environment
   $ conda activate httomolib # activate the environment
   $ pip install httomolib

Setup the development environment:
==================================

.. code-block:: console
    
   $ git clone git@github.com:DiamondLightSource/httomolib.git # clone the repo
   $ conda create --name httomolib # create a fresh conda environment
   $ conda activate httomolib # activate the environment
   $ pip install -e . # development mode
