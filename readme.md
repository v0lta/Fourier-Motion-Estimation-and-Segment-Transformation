Fourier Motion Estimation and Image Transformation Source code
--------------------------------------------------------------------------------------------------------------------

This repository contains the source code for the paper Object-centered Fourier Motion Estimation
and Image Transformation for Video Prediction [ESANN 2020, to appear]. Please take a look at the 
paper to learn more about the underlying math.

##### Image registration
Pytorch_registration module 'util/pytorch_registration.py' contains code to estimate image translation and rotation.
The pytorch implementation is based on the numpy code at https://www.lfd.uci.edu/~gohlke/code/imreg.py.html. 

##### Image transformation
The 'util/rotation_translation_pytorch.py' module in ships my implementation of the three pass frequency domain
approach to image translation and rotation, which inspired by the description in 
https://tspace.library.utoronto.ca/bitstream/1807/11762/1/MQ28850.pdf.
I would like to thank Hafez Farazi for helping me debug the log-polar transformation pytorch code.

#### RNN cell
The correction network is implemented in 'cell/registration_cell.py' 

#### Reproduction
To reproduce results from the paper run 'train_reg_gru.py'.

#### Dependencies
This project has been developed using pytorch version 1.4.0 and Tensorboard 2.1.0 on Nvidia Titan Xp cards.

#### Known Issues
The rotations are currently measured with respect to the image center. Working with the object center will 
simplify things. Translation and rotation registration is limited to pixel level accuracy.