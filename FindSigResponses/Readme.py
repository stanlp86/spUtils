"""

A lot of the parsing that this will require can be done with 

8.3.3.1. defaultdict Examples
https://docs.python.org/2/library/collections.html#collections.defaultdict



We want to pass a time x cells x trials array along with a dictionary containing odor timing and sequence information. 


We want the flexibility to monitor either suppression or excitation.

We want to flexibly specify periods in the time-series to be used for statistical comparisons 
	-while keeping either odor, 
	-or cell stationary. 

We need a distribution of activity expected from chance sampling to which to compare a response.
	-To make it, we need a random generator that will sample with replacement and return contiguous periods in the baseline. 



We need a generator to premake this dictionary. 
We need a generator to fill a dictionary that can be accessed in a cell-centric or odor-centric way. 
	-this dictionary must be filled by siultaneosly updating multiple keys. Use defaultdict see below. 


The methods for proving the input, assessing a response , and storing the output should be built independently and then  wrapped together. 





"""
"""
100 cells 23 odors 10 trials

most efficeient way to store 23000 samples?
we can always swap key value ordering with dict comprehension.
 >>> def invert(d):
    ...     return {v : k for k, v in d.iteritems()}

    or 

    from collections import defaultdict

	output = defaultdict(dict)

	for (key, user), value in my_dict.iteritems():
	    output[user][key] = value

"""