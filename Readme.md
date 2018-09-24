This project combines Model-based (MBD) and Data-driven (DDD) methods to diagnose hybrid systems (HS).

# Framework
HS ---> Simple HSs ---> normal Hybrid Estimation(MBD) ---> consistent? ---> Normal or Identified Faults
                                ^                               |       Yes
                                |                               | No
                                |                               |
                         Possible Faults <-- -Classifier(DDD)<---

The original hybrid system is deomposed into several simple hybrid systems. A MBD method is used to track
the states. If an inconsistency is found, there must be a fault. A classifier (DDD) is employed to estimate
which fault it is. Then the estimator is used again to verify the results given by the classifier. At last,
a consistent diagnosis result is given.

# Minimal Structrural Overdetermined Set (MSO) and Binary Integer Programming (BIP) based HS decomposition



# Search and Confidency based Hybrid Estimation



# Neural Network based Classifier

