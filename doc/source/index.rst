.. themis-ml documentation master file, created by
   sphinx-quickstart on Sat Sep 23 16:23:06 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: _static/themis-ml-banner.png

---------------------------------------------

A fairness-aware machine learning library
=========================================

**themis-ml** is an open source machine learning library that implements
several fairness-aware methods that comply with the sklearn API.

Fairness-aware Machine Learning
===============================

**themis-ml** defines discrimination as the preference (bias) for or against a
set of social groups that result in the unfair treatment of its members with
respect to some outcome.

It defines fairness as the opposite of discrimination, and in the context of a
machine learning algorithm, this is measured by the degree to which the
algorithm's predictions favor one social group over another in relation to an
outcome that holds socioeconomic, political, or legal importance, e.g. the
denial/approval of a loan application.

An algorithm is "fair" depending on how we define fairness, the outcome of
interest, and the socially sensitive attributes that relate to potentially
discriminatory circumstances. For example, if we consider fairness as
statistical parity, a fair algorithm is one in which the proportion of approved
loans among minorities is equal to the proportion of approved loans among white
people.

However, there are many other ways to define and operationalize fairness, and
the purpose of **themis-ml** is to attempt to provide an interface that gives
users with access to formalized definitions of fairness and discrimination
described in the  the machine learning and statistics literature. Check out this
`paper <https://github.com/cosmicBboy/themis-ml/blob/master/paper/main.pdf>`_
for more details.

Install
=======

You can install `themis-ml` with `conda` or `pip`. Currently only
Python 2.7 and 3.6 are supported.

.. code-block:: python

    # conda
    conda install themis-ml

    # pip
    pip install themis-ml


.. toctree::
   :maxdepth: 4
   :caption: Contents

   API

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
