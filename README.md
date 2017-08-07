# Mapping Patient Trajectories using Longitudinal Extraction and Deep Learning in the MIMIC-III Critical Care Database:

Brett K. Beaulieu-Jones <sup>1</sup>, Patryk Orzechowski <sup>1</sup>  and Jason H. Moore <sup>1</sup>

<sup>1</sup> Genomics and Computational Biology Graduate Group, Perelman School of Medicine, University of Pennsylvania, Philadelphia, Pennsylvania, USA.

*To whom correspondence should be addressed: (brettbe) at med.upenn.edu

Introduction
--------
Electronic Health Records (EHRs) contain a wealth of patient data useful to biomedical researchers. At present, both the extraction of data and methods for analyses are frequently designed to work with a single snapshot of a patient’s record. Health care providers often perform and record actions in small batches over time. By extracting these encounters, a sequence can be formed providing a trajectory for a patient’s interactions with the health care system. These encounters also offer a basic heuristic for the level of attention a patient receives from health care providers. We show that is possible to learn meaningful embeddings from these encounters using two deep learning techniques, unsupervised autoencoders and long short-term memory networks. We compare these methods to traditional machine learning methods which require a point in time snapshot to be extracted from an EHR.


Feedback
--------

Please feel free to email us - (brettbe) at med.upenn.edu with any feedback or
raise a github issue with any comments or questions.

Acknowledgements
----------------

We thank Casey S. Greene (University of Pennsylvania) for his helpful discussions. Funding: This work was supported by the Commonwealth Universal Research Enhancement (CURE) Program grant from the Pennsylvania Department of Health. B.K.B.-J. and J.H.M. were also supported by  and by US National Institutes of Health grants AI116794 and LM010098 to J.H.M..
