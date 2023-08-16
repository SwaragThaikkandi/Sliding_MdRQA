# Implementing Sliding Window MdRQA to get Summary Statistics Estimate of MdRQA measures from the Data
This is a brief tutorial about how to use the functions provided in the github repository. Here, we are providing codes, using which the RQA measures can be estimated. However, we uses a parameter search for finding the embedding dimension. This is a step by step tutorial
## import packahges 
We will begin with importing packages and functions. Note that the python files should be copied to the mail analysis directory
```python
import numpy as np
from scipy.integrate import solve_ivp
import operator
import contextlib
import functools
import operator
import warnings
from numpy.core import overrides
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pandas as pd
from os import listdir
from os.path import isfile, join
from collections import defaultdict
import csv
from tqdm import tqdm
import os
import numpy as np
import operator
import contextlib
import functools
import operator
import warnings
from numpy.core import overrides
import matplotlib.pyplot as plt
import pandas as pd
from os import listdir
from os.path import isfile, join
from collections import defaultdict
import csv
from tqdm import tqdm
import pickle
import random
from scipy.stats import skew
from p_tqdm import p_map
from functools import partial
from scipy.interpolate import pchip_interpolate
import memory_profiler
import ast

from kuramoto import Kuramoto, plot_phase_coherence, plot_activity # see https://github.com/fabridamicelli/kuramoto
# For installing the kuramoto package run: pip install kuramoto

from RP_maker import RP_computer

from Extract_from_RP import Mode
from Extract_from_RP import Check_Int_Array
from Extract_from_RP import Sliding_window
from Extract_from_RP import Whole_window
from Extract_from_RP import windowed_RP
from Extract_from_RP import First_middle_last_avg
from Extract_from_RP import First_middle_last_sliding_windows
from Extract_from_RP import First_middle_last_sliding_windows_all_vars

from cross_validation import feature_selection
from cross_validation import nested_cv
```
## Example 2: Kuramoto Model
In this example, we will be simulating the Kuramoto model, varying number of scillators, length of time series, and the coupling strength. The number of oscillators will be randomly choosen from a descrete uniform distribution of integers from 3 to 6. Time series length is also choosen similarly. Coupling strength is sampled from a continuous uniform distribution from 0 to 2Kc, where Kc is the critical coupling strength. 
We are using the mean field Kuramoto model, where the coupling strength between any two oscillator is the same. The system is given by the differential equation: 

$$\dot{\theta_{i}} = \omega_{i} + \sum_{j=1}^{N} K_{ij} \sin{(\theta_{j}-\theta{i})}, i=1, ..., N$$


We will sample the frequencies($$\omega_{i}$$) from a standard normal distribution. And the critical coupling strength for mean field model is given by:


$$K_{c} = |\omega_{max}-\omega_{min}|$$


And in the case of mean field model, the coupling strength is given by: 


$$K_{ij} = K/N >0, \forall i,j \in \{1,2,3,...,N\}$$


Here, synchrony of the system is defined in terms of a complex valued order parameter($$r$$), which is given by:


$$r e^{i\psi} =  \frac{1}{N} \sum_{j=1}^{N}e^{i \theta_{j}}$$

Here $$\psi$$ is the average phase value. To arrive at  an expression that makes the dependence of synchrony on values of K explicit, we begin with multiplying both sides by $$e^{-i \theta_{i}}$$.

$$r e^{i\psi} e^{-i \theta_{i}} = \left( \frac{1}{N} \sum_{j=1}^{N}e^{i \theta_{j}}\right) e^{-i \theta_{i}}$$

$$r e^{i(\psi -\theta_{i})}= \frac{1}{N} \sum_{j=1}^{N}e^{i(\theta_{j}-\theta_{i})}$$

$$\dot{\theta_{i}} = \omega_{i} + K r \sin{(\psi -\theta_{i})}$$

Here, when the coupling strength is tending to zero, the oscillators would be oscillating in their natural frequencies. 
### What is the order parameter? How does it matter?
<iframe src="https://github.com/SwaragThaikkandi/Sliding_MdRQA/blob/main/Fig14.pdf" width="100%" height="500" 
