===================================
Generic and Open Learning Federator
===================================


.. image:: https://img.shields.io/pypi/v/golf_federated.svg
        :target: https://pypi.python.org/pypi/golf_federated
        :alt: PyPI Version

.. image:: https://readthedocs.org/projects/generic-and-open-learning-federator/badge/?version=latest
        :target: https://generic-and-open-learning-federator.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

.. image:: https://app.travis-ci.com/IntelligentSystemsLab/generic_and_open_learning_federator.svg?token=uyV9JpsFqExQVbjDeQ5q&branch=main
        :alt: Build Status




A scalable, portable, and lightweight Federated Learning framework.


* **Free software**: MIT license
* **Documentation**: https://generic-and-open-learning-federator.readthedocs.io.



Features
--------

* GOLF provides a lightweight solution to support the implementation of FL.
* GOLF modularizes system functions to achieve loose coupling during system development and deployment, which makes the framework more generic and scalable.
* GOLF uses container technology to ensure that the system is weakly dependent on the compilation environment to achieve portability.
* GOLF is compatible with multiple devices (e.g., Android, embedded computers, edge devices, etc.).

News
--------

#. 🌟 **June 07, 2024** - Introducing Cedar:

  Cedar is a secure, cost-efficient, and domain-adaptive framework for federated meta-learning. Key features include:

  - 💡 **Federated Meta-Learning**: Enable a safeguarded knowledge transfer with high model generalizability and adaptability.
  - 📨 **Cost-Efficient**: Implement a layer-wise model uploading mechanism to reduce communication cost.
  - 🔒 **Robust Security**: Defend against malicious attacks like data inversion and model poisoning.
  - 🔧 **High Performance**: Support high-performance personalization and customization of globally shareable meta-models.

Installation
-------------

To install GOLF, simply use pip:

.. code-block:: sh

    pip install golf_federated

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage


Contributing
------------

We welcome contributions! Here are some ways you can help:

1. Report bugs and request features on GitHub Issues: https://github.com/IntelligentSystemsLab/generic_and_open_learning_federator/issues
2. Submit pull requests to improve the codebase.

Contact
-------

For any questions or issues, please contact the development team at `guozh29@mail2.sysu.edu.cn`.

Related Publications
--------------------

We have published several papers related to this project:

    [1]  `L. You, S. Liu, B. Zuo, C. Yuen*, D. Niyato, H. V. Poor,"Federated and Asynchronized Learning for Autonomous and Intelligent Things", IEEE Network Magazine, 2023. <https://ieeexplore.ieee.org/document/10274563>`_

    [2]  `L. You, S. Liu, T. Wang, B. Zuo, Y. Chang, C. Yuen*,"AiFed: An Adaptive and Integrated Mechanism for Asynchronous Federated Data Mining", IEEE Transactions on Knowledge and Data Engineering, 2023. <https://ieeexplore.ieee.org/document/10316646>`_

    [3]  `L. You, Z. Guo, B. Zuo, Y. Chang*, C. Yuen,"SLMFed: A Stage-based and Layer-wise Mechanism for Incremental Federated Learning to Assist Dynamic and Ubiquitous IoT", IEEE Internet of Things Journal, 2024. <https://ieeexplore.ieee.org/document/10399971>`_

    [4]  `L. You, S. Liu, Y. Chang, C. Yuen*,"A triple-step asynchronous federated learning mechanism for client activation, interaction optimization, and aggregation enhancement", IEEE Internet of Things Journal, 2022. <https://ieeexplore.ieee.org/document/9815310>`_

    [5]  `S. Liu, L. You*, R. Zhu, B. Liu, R. Liu, Y. Han, C. Yuen,"AFM3D: An Asynchronous Federated Meta-learning Framework for Driver Distraction Detection", IEEE Transactions on Intelligent Transportation Systems, 2024. <https://ieeexplore.ieee.org/document/10423999>`_
