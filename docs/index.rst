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

* 1) **Aggregation Algorithm**: Most existing frameworks only support synchronous FL, but asynchronous FL is more suitable for applications with heterogeneous learning participants.
* 2) **Device Compatibility**: FL framework needs to be capable of hosting various devices, especially edge devices, and performing on-device computing.
* 3) **API Design**: Generic calls and flexible expansion of APIs are the essential requirements for an efficient and scalable FL framework.
* 4) **System Configuration**: Users and developers prefer out-of-the-box system frameworks with reproducible procedures.
* 5) **System Application**: The system should provide service customization to support various application scenarios.

GOLF is a possible solution for these challenges and is constantly being improved. You can get a quick start with :doc:`start`.


Feature
------------

* 1) GOLF provides a lightweight solution to support the implementation of FL.
* 2) GOLF modularizes system functions to achieve loose coupling during system development and deployment, which makes the framework more generic and scalable.
* 3) GOLF uses container technology to ensure that the system is weakly dependent on the compilation environment to achieve portability.
* 4) GOLF is compatible with multiple devices (e.g., Android, embedded computers, edge devices, etc.).

.. toctree::
   :maxdepth: 2
   :caption: Outline

   start
   configuration
   api
   contributor


Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. Collaborators
.. ==================

|sysu|

|islab| |jysw|

|mit| |sutd| 

.. |islab| image:: ./source/is-lab.png
    :height: 100px
    :width: 100px

.. |jysw| image:: ./source/jyswlogo.png
    :height: 100px
    :width: 100px

.. |mit| image:: ./source/mit.png
    :height: 100px
    :width: 100px

.. |sutd| image:: ./source/sutd.jpg
    :height: 100px
    :width: 100px
    
.. |sysu| image:: ./source/sysu.jpg
    :height: 100px
    :width: 100px