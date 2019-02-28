This project combines Model-based (MBD) and Data-driven (DDD) methods to diagnose hybrid systems (HS).

# Framework

```
                -----------------------------------------------------------------
                |                                                               |
                V                                                               |
Obs -----> |Particle Filter| --->  Res ---> |ANN Fault Identifier| ---> |Variance Analysis| 
 |              ^                   |                                           ^
 |              |                   V                                           |
 --------> |ANN Mode Detector|   |ANN Fault Isolator| --------------------------|
 ```

ANN mode detector employs Obs to detect the current mode. Particle filter estimate the current continuous state based on 
the discrete mode and Obs to generate residual Res. ANN fault isolator and ann fault identifier utilize Res to detect and 
isolate if there is a fault and what is the fault magnitude is. If there is no fault, the output of fault identifier will
be ignored because the output may not be zero, although should a small number. When a fault is detected by the isolator, 
the corresponding value given by the fault identifier is analyzed by variance analysis. If the estimated fault magnitudes 
in the last two windows are sampled from the same distribtuion, the averge value of the outputs in the two windows is computed 
as the final estimated magnitude. The corresponding parameter is the model is replaced and fault detector is closed, because 
we assume there is only one fault.

# Particle Filter
The details about PF please refers to corresponding papers. It is worthy noting that in normalization, if the maximal weight 
in the particles is a small number, normaliztion should give up the likelihood and reset all the particle with an equal weight. 
Because in this case, a fault occurs and the likelihood can not be used to update the continuous state. Ignoring the likelihood 
means we just similate the model in its normal behavior and the residuals are the same with what we used in training.

# ANN Mode Detector
Obs ---> CNN ---> GRU ---> FC ---> label distribution
# ANN Fault Isolator
Res ---> CNN ---> GRU ---> FC ---> magnitude
# ANN Fault Identifier
Res ---> CNN ---> GRU ---> FC ---> magnitude
# Variance Analysis
Test if the estimated magnitudes in the last two windows come from the same distribution.
