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

Citations
--------

If this project is helpful to your research, please cite our papers:

  `L. You, Z. Guo, C. Yuen*, C.Y.C. Chen, Y. Zhang, H.V. Poor,"A framework reforming personalized Internet of Things by federated meta-learning", Nature Communications, 2025. <https://www.nature.com/articles/s41467-025-59217-z>`_

  `L. You, Z. Guo, B. Zuo, Y. Chang*, C. Yuen,"SLMFed: A Stage-based and Layer-wise Mechanism for Incremental Federated Learning to Assist Dynamic and Ubiquitous IoT", IEEE Internet of Things Journal, 2024. <https://ieeexplore.ieee.org/document/10399971>`_

  `S. Liu, L. You*, R. Zhu, B. Liu, R. Liu, Y. Han, C. Yuen,"AFM3D: An Asynchronous Federated Meta-learning Framework for Driver Distraction Detection", IEEE Transactions on Intelligent Transportation Systems, 2024. <https://ieeexplore.ieee.org/document/10423999>`_

  `L. You, S. Liu, B. Zuo, C. Yuen*, D. Niyato, H. V. Poor,"Federated and Asynchronized Learning for Autonomous and Intelligent Things", IEEE Network Magazine, 2023. <https://ieeexplore.ieee.org/document/10274563>`_

  `L. You, S. Liu, T. Wang, B. Zuo, Y. Chang, C. Yuen*,"AiFed: An Adaptive and Integrated Mechanism for Asynchronous Federated Data Mining", IEEE Transactions on Knowledge and Data Engineering, 2023. <https://ieeexplore.ieee.org/document/10316646>`_

  `L. You, S. Liu, Y. Chang, C. Yuen*,"A triple-step asynchronous federated learning mechanism for client activation, interaction optimization, and aggregation enhancement", IEEE Internet of Things Journal, 2022. <https://ieeexplore.ieee.org/document/9815310>`_

.. code-block:: sh

    @article{You2025framework,
      title={A framework reforming personalized Internet of Things by federated meta-learning},
      author={You, Linlin and Guo, Zihan and Yuen, Chau and Chen, Calvin Yu-Chian and Zhang, Yan and Poor, H. Vincent},
      journal={Nature communications},
      volume={16},
      pages={3739},
      year={2025},
      publisher={Nature Publishing Group UK London}
    }

    @article{you2024slmfed,
      title={SLMFed: A Stage-Based and Layerwise Mechanism for Incremental Federated Learning to Assist Dynamic and Ubiquitous IoT},
      author={You, Linlin and Guo, Zihan and Zuo, Bingran and Chang, Yi and Yuen, Chau},
      journal={IEEE Internet of Things Journal},
      volume={11},
      number={9},
      pages={16364--16381},
      year={2024},
      publisher={IEEE}
    }  

    @article{liu2024afm3d,
      title={AFM3D: An asynchronous federated meta-learning framework for driver distraction detection},
      author={Liu, Sheng and You, Linlin and Zhu, Rui and Liu, Bing and Liu, Rui and Yu, Han and Yuen, Chau},
      journal={IEEE Transactions on Intelligent Transportation Systems},
      volume={25},
      number={8},
      pages={9659--9674},
      year={2024},
      publisher={IEEE}
    }

    @article{you2023federated,
      title={Federated and asynchronized learning for autonomous and intelligent things},
      author={You, Linlin and Liu, Sheng and Zuo, Bingran and Yuen, Chau and Niyato, Dusit and Poor, H Vincent},
      journal={IEEE Network},
      volume={38},
      number={2},
      pages={286--293},
      year={2023},
      publisher={IEEE}
    }

    @article{you2023aifed,
      title={AiFed: An adaptive and integrated mechanism for asynchronous federated data mining},
      author={You, Linlin and Liu, Sheng and Wang, Tao and Zuo, Bingran and Chang, Yi and Yuen, Chau},
      journal={IEEE Transactions on Knowledge and Data Engineering},
      volume={36},
      number={9},
      pages={4411--4427},
      year={2023},
      publisher={IEEE}
    }

    @article{you2022triple,
      title={A triple-step asynchronous federated learning mechanism for client activation, interaction optimization, and aggregation enhancement},
      author={You, Linlin and Liu, Sheng and Chang, Yi and Yuen, Chau},
      journal={IEEE Internet of Things Journal},
      volume={9},
      number={23},
      pages={24199--24211},
      year={2022},
      publisher={IEEE}
    }

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

