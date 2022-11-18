# membraneProtML
Description:
ML application capable of identifying if a amino acid sequence is a transmembrane protein.

This project essentially serves as a sandbox where I test out new ML architectures as I learn them in order to get a 
better understanding of how they are implemented.  Due to the haphazard way I've constructed this project, it is not
a great representation of my coding practices as much as a living record of my learning.  Feel free to check out the 
rest of my github if you are interested in more examples of my work :)

------------------------------------------------------------------------------------------------------------------------

#####Models implemented thus far

Absolute Naive Solution:
Counts up all the prevelence of all the residues in the sequence, and trains an SVM classifier.  Only works due to 
the simple nature of this classification problem, will fail if it gets any more complex.

Kernel SVM solution:
Implement the kernel suggested in this(https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-017-1560-9) 
paper.  This one works better than guessing, but calculating the kernel is obscenely slow, even after a 10x speedup 
thanks to dynamic programming implementation. 

Transformer Implementation (work in progress):
This is my first implementation of a transformer architecture, and it is still under construction.
