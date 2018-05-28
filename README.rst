nlputils
========

This repository contains several functions to analyze text corpora.
Mainly, text documents can be transformed into (sparse, dictionary based) tf-idf features, based on which the similarities between the documents can be computed, the dataset can be classified with knn, or the corpus can be visualized in two dimensions. 

The individual library components are largely independent of another (besides most of them using functions from ``dict_utils.py``), which means you might also find only parts of this library interesting, e.g. ``embedding.py``, which contains a concise python implementation of t-SNE, which can be used to embed data points in 2D based on any kind of similarity matrix, not necessarily created with the scripts from this library.

If any of this code was helpful for your research, please consider citing it:

.. image:: https://zenodo.org/badge/17917498.svg
   :target: https://zenodo.org/badge/latestdoi/17917498

::

    @misc{franziska_horn_2018_1254413,
      author       = {Franziska Horn},
      title        = {cod3licious/nlputils},
      month        = may,
      year         = 2018,
      doi          = {10.5281/zenodo.1254413},
      url          = {https://doi.org/10.5281/zenodo.1254413}
    }


The code is intended for research purposes. It was programmed for Python 2.7, but should also run on newer Python 3 versions - please open an issue if you find something isn't working there!


installation
------------
You either download the code from here and include the nlputils folder in your ``$PYTHONPATH`` or install (the library components only) via pip:

    ``$ pip install nlputils``

nlputils library components
---------------------------

dependencies: numpy, scipy, unidecode, matplotlib

- ``dict_utils.py``: various helper functions to manipulate dictionaries, e.g. to invert them on various levels (for example transform a dict with ``{document: {word: count}}`` into ``{word: {document: count}}``).
- ``features.py``: this contains code to preprocess texts and transform them into tf-idf features. It's somewhat similar to the sklearn TfidfVectorizer, but based on (sparse) dictionaries instead of sparse vectors. These dictionary based document features are the main input used for other parts of this library. But there is also a ``features2mat`` function to transform the dictionaries into a sparse feature matrix, which can be used with sklearn classifiers, for example.
- ``simcoefs.py``: this has one main function ``compute_sim``, which gets as input the tf-idf feature dictionaries of two documents and then computes their similarity. Concerning the type of similarity to compute between the documents, you can chose from a large variety of similarity coefficients, kernel functions, and distance measures, implemented based on [RIE08]_.
- ``simmat.py``: this contains wrapper functions for ``simcoefs.py`` to speed up the computation of the similarity matrix for a whole corpus.
- ``ml_utils.py``: helper function to perform a cross-validation.
- ``knn_classifier.py``: based on a similarity matrix, perform k-nearest-neighbors classification.
- ``embedding.py``: based on a similarity matrix, project data points to 2D with classical scaling or t-SNE.
- ``visualize.py``: helper functions to create a plot of the dataset based on the 2D embedding. This can also create a json file, which can be used with d3.js to create an interactive visualization of the data.

examples
--------

additional dependencies: sklearn

In the iPython Notebook at |examples/examples.ipynb|_ are several examples on how to use the above described library components.

.. |examples/examples.ipynb| replace:: ``examples/examples.ipynb``
.. _examples/examples.ipynb: https://github.com/cod3licious/nlputils/blob/master/examples/examples.ipynb

If you have any questions please don't hesitate to send me an `email <mailto:cod3licious@gmail.com>`_ and of course if you should find any bugs or want to contribute other improvements, pull requests are very welcome!

.. [RIE08] Rieck, Konrad, and Pavel Laskov. "Linear-time computation of similarity measures for sequential data." *Journal of Machine Learning Research* 9.Jan (2008): 23-48.
