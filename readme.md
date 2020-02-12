Fourier Motion Estimation and Image Transformation Source code
--------------------------------------------------------------------------------------------------------------------

This repository contains the source code for the paper 
Object-centered Fourier Motion Estimation and Image Transformation for Video Prediction [ESANN 2020, to appear]

The 'util' folder contains pytorch tools for frequency domain image registration and manipulation. The
pytorch_registration module containes a partial pytorch port of the registration code available at
https://www.lfd.uci.edu/~gohlke/code/imreg.py.html . 
The 'rotation_translation_pytorch' module in 'util' ships my implementation of the three pass frequency domain
approach to image translation and rotation, which inspired by the description in 
https://tspace.library.utoronto.ca/bitstream/1807/11762/1/MQ28850.pdf . Please read the paper for more details.
I would like to thank Hafez Farazi for helping my debugging the log-polar transformation pytorch code.

To reproduce results from the paper run 'train_reg_gru.py'.