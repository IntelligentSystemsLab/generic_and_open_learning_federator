.. golf_federated documentation master file, created by
   sphinx-quickstart on Sat Nov 19 22:35:40 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GOLF!
==========================================

GOLF is a general and open-source Federated Learning (FL) framework. Through this document, you can learn everything you need to know when using GOLF. 


What it is
------------

GOLF (Generic and Open Learning Federator) aims to present a scalable, portable, and lightweight framework, which can be applied to diverse devices
in different scenarios. 

For FL frameworks, the following five aspects summarize the
challenges in their design, implementation, and other processes:

* Aggregation Algorithm: Most existing frameworks only support synchronous FL, but asynchronous FL is more suitable for applications with heterogeneous learning participants.
* Device Compatibility: FL framework needs to be capable of hosting various devices, especially edge devices, and performing on-device computing.
* API Design: Generic calls and flexible expansion of APIs are the essential requirements for an efficient and scalable FL framework.
* System Configuration: Users and developers prefer out-of-the-box system frameworks with reproducible procedures.
* System Application: The system should provide service customization to support various application scenarios.

GOLF is a possible solution for these challenges and is constantly being improved. You can get a quick start with :doc:`start`.


Feature
------------

* GOLF provides a lightweight solution to support the implementation of FL.
* GOLF modularizes system functions to achieve loose coupling during system development and deployment, which makes the framework more generic and scalable.
* GOLF uses container technology to ensure that the system is weakly dependent on the compilation environment to achieve portability.
* GOLF is compatible with multiple devices (e.g., Android, embedded computers, edge devices, etc.).

.. The remainder of this paper is organized as follows. Section 2
.. surveys related work and summarizes them based on
.. the above challenges. Section 3 introduces the design and
.. implementation of GOLF. Section 4 performs some experiments
.. based on GOLF. Section 5 draws a conclusion and outlines
.. future work.

.. +-----------------------+---------------------+------+--------+-------+---------+--------+--------+--------+-----+-----+------+-------+------+
.. |        Category       |    Specific Issue   | Fate | PySyft | APPFL | OpenFed | OpenFL | FLOWER | FEDLAB | TFF | PFL | FEDn | FedML | GOLF |
.. +=======================+=====================+======+========+=======+=========+========+========+========+=====+=====+======+=======+======+
.. | Aggregation Algorithm |     Synchronous     |   ✓  |    ✓   |   ✓   |    ✓    |    ✓   |    ✓   |    ✓   |  ✓  |  ✓  |   ✓  |   ✓   |   ✓  |
.. |                       +---------------------+------+--------+-------+---------+--------+--------+--------+-----+-----+------+-------+------+
.. |                       |     Asynchronous    |   ✓  |    ✓   |       |         |        |    ✓   |    ✓   |     |     |      |       |   ✓  |
.. +-----------------------+---------------------+------+--------+-------+---------+--------+--------+--------+-----+-----+------+-------+------+
.. |  Device Compatibility |   Device Diversity  |   ✓  |        |       |         |        |        |        |     |  ✓  |   ✓  |   ✓   |   ✓  |
.. |                       +---------------------+------+--------+-------+---------+--------+--------+--------+-----+-----+------+-------+------+
.. |                       | On-device Computing |      |        |       |         |        |    ✓   |        |     |     |      |   ✓   |   ✓  |
.. +-----------------------+---------------------+------+--------+-------+---------+--------+--------+--------+-----+-----+------+-------+------+
.. |       API Design      |     Generic Call    |      |        |   ✓   |         |        |    ✓   |        |     |     |   ✓  |   ✓   |   ✓  |
.. |                       +---------------------+------+--------+-------+---------+--------+--------+--------+-----+-----+------+-------+------+
.. |                       |  Flexible Expansion |      |        |   ✓   |    ✓    |        |    ✓   |    ✓   |     |     |   ✓  |   ✓   |   ✓  |
.. +-----------------------+---------------------+------+--------+-------+---------+--------+--------+--------+-----+-----+------+-------+------+
.. |  System Configuration |    Out of the box   |      |    ✓   |   ✓   |    ✓    |    ✓   |        |    ✓   |  ✓  |     |   ✓  |       |   ✓  |
.. |                       +---------------------+------+--------+-------+---------+--------+--------+--------+-----+-----+------+-------+------+
.. |                       |     Reproducible    |      |    ✓   |       |         |        |        |        |     |     |   ✓  |       |   ✓  |
.. |                       |     Environments    |      |        |       |         |        |        |        |     |     |      |       |      |
.. +-----------------------+---------------------+------+--------+-------+---------+--------+--------+--------+-----+-----+------+-------+------+
.. |   System Application  |     Unrestricted    |   ✓  |        |       |    ✓    |        |    ✓   |    ✓   |     |  ✓  |   ✓  |   ✓   |   ✓  |
.. |                       |      Scenarios      |      |        |       |         |        |        |        |     |     |      |       |      |
.. |                       +---------------------+------+--------+-------+---------+--------+--------+--------+-----+-----+------+-------+------+
.. |                       |       Service       |      |        |       |         |        |        |    ✓   |     |     |      |       |   ✓  |
.. |                       |    Customization    |      |        |       |         |        |        |        |     |     |      |       |      |
.. +-----------------------+---------------------+------+--------+-------+---------+--------+--------+--------+-----+-----+------+-------+------+


.. toctree::
   :maxdepth: 2
   :caption: Outline

   start
   configuration
   api


Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
