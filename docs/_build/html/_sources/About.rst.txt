.. _About:
  
***************
About
***************
Background
-----------

The easyQuake package combines earthquake waveform download, event detection at individual stations, event association, magnitude determination, and absolute location with hypoinverse.

Each module in the easyQuake package is called, individually, with a driver script and we include several example driver scripts in these docs and in the Github repository: https://github.com/jakewalter/easyQuake

The easyQuake platform utilizes a choice between the GPD picker (Ross et al., 2019) and the EQTransformer (Mousavi et al., 2020) deep-learning pickers and easyQuake will utilize the default models for those pickers. However, in most circumstances, you may want to train your own picker if you have a sufficient dataset for your experiment or region of interest.

Detection Improvement
----------------------

At OGS, we run the seismic network and create scientific products (location, magnitude, etc.) that are released through USGS as part of our membership in the national Advanced National Seismic System (ANSS). In adding easyQuake to augment detection, we have found a factor of 2 more earthquakes since May 2020.

.. image:: _static/oklahoma_case.png
  :width: 800
  :alt: Oklahoma seismicity detection improved

In addition, as a test scenario, we can run it on FDSN-downloaded waveforms and find significant detection improvementrelative to the regional seismic network in the case of the March 2020 M6.5 Central Idaho eartquake

.. image:: _static/idaho.png
  :width: 800
  :alt: Idaho detection example

In Development
---------------
We have some tailored some of the standard easyQuake modules to large-N node datasets and other special applications. Those will be available at the next major release. Also, we have a quasi-realtime detection workflow, utilizing easyQuake on short snippets of data gathered from a seedlink stream, that is currently being tested.

References
-----------
* Ross, Z. E., M.-A. Meier, E. Hauksson, and T. H. Heaton (2018), Generalized seismic phase detection with deep learning, Bull. Seismol. Soc. Am., 108, doi: 10.1785/0120180080.

* Mousavi, S.M., Ellsworth, W.L., Zhu, W., Chuang, L.Y., Beroza, G.C. (2020), Earthquake Transformer: An Attentive Deep-learning Model for Simultaneous Earthquake Detection and Phase Picking, Nature Communications

* Walter, J. I., P. Ogwari, A. Thiel, F. Ferrer, and I. Woelfel (2021), easyQuake: Putting machine learning to work for your regional seismic network or local earthquake study, Seismological Research Letters, 92(1): 555–563, https://doi.org/10.1785/0220200226.
