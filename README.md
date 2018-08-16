# openLBMPM
openLBMPM is an open source lattice Boltzmann method (LBM) package for multicomponent and multiphase (MCMP) flow and transport in porous media. Currently, it includes Shan-Chen method and color gradient method for MCMP system. Currently, the transport part does not include any chemical reactions and phase change, but they will be added later. 

D2Q9 and D3Q19 schemes are implemented to simulate fluid flow in 2D and 3D. To balance the accuracy and efficiency, D2Q5 and D3Q7 schemes are used to simulate the transport phenomena. 

There are two options for Shan-Chen method: (1) Original Shan-Chen method, which integrates the force term to the equilibrium velocity and cannot reach high viscosity ratio; (2) Explicit forcing model developed by M.Porter et al (2012), which is able to reach high viscosity ratio with the different isotropy values  (M~1000). The last one has multi-relaxation-time collision operators to suppress spurious currents and keep numerically stable. For color gradient model, the methods developed by Liu et.al (2014), Huang et al (2014) and Takashi et al (2018) are included here. 
#Required packages
OpenLBMPM is accelerated by GPU parallel computation, so it needs several supporting packages:

1. [CUDA](https://developer.nvidia.com/cuda-downloads)
2. [Anaconda](https://www.anaconda.com/download/#linux)
