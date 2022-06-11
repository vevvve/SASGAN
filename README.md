Python implementation code for the paper titled,

Title: 3D super-resolution reconstruction of porous media based on GANs and CBAMs

Authors: Ting Zhang1, Qingyang Liu1, Yi Du2, * 

1.College of Computer Science and Technology, Shanghai University of Electric Power, Shanghai 200090, China 

2.College of Engineering, Shanghai Polytechnic University, Shanghai 201209, China

(*corresponding author, E-mail: duyi0701@126.com. Tel.: 86 - 21- 50214252. Fax: 86 - 21- 50214252. )

Ting Zhang Email: tingzh@shiep.edu.cn, Affiliation: College of Computer Science and Technology, Shanghai University of Electric Power, Shanghai 200090, China

Qingyang Liu: ve.vvve@mail.shiep.edu.cn, Affiliation: College of Computer Science and Technology, Shanghai University of Electric Power, Shanghai 200090, China

Yi Du E-mail: duyi0701@126.com, Affiliation: College of Engineering, Shanghai Polytechnic University, Shanghai 201209, China

# SASGAN


1.requirements

Pytorch == 1.7.0

To run the code, an NVIDIA GeForce RTX3080 GPU video card with 10GB video memory is required. 

Software development environment should be any Python integrated development environment used on an NVIDIA video card. 

Programming language: Python 3.7.9. 


2.How to use？


First, preprocess the image: Cut the porous media slice into 80 * 80 * 80 size pictures and combine into one tif. Then run the lowres.py to get LR tif as the training image.


Secondly, set the network parameters such as maxscalenum, batchsize, learning rate and storage location.  After configuring the parameters and environment, you can run maim.py directly.


