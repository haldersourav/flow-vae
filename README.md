# Flow-VAE: A variational autoencoder (VAE) based extraction of various characteristics of flow through an active collapsible tube
Balloon dilation catheters are often used to quantify the physiological state of peristaltic activity in tubular organs (such as the esophagus). This work is aimed to investigate the effect of a solitary peristaltic wave on a fluid-filled elastic tube with closed ends. A physics model calculates the resulting tube wall deformations, flow velocities, and pressure variations by solving 1D mass and momentum conservation equations along with a linear pressure tube law. The major variables that contribute to the different flow regimes inside the deformable tube depend on the fluid viscosity, wall stiffness, contraction strength, and active relaxation of the tube walls. 

The VAE takes the tube geometry as input and generates a latent space that captures the underlying similarities and dissimilarities between various flow regimes as well as identifies the underlying effect of the variables that cause the different regimes. More details of this work can be found in the following paper:

Halder, S., Yamasaki, J., Acharya, S., Kou, W., Elisha, G., Carlson, D. A., Kahrilas, P. J., Pandolfino, J. E., & Patankar, N. A. (2022). Virtual disease landscape using mechanics-informed machine learning: Application to esophageal disorders. Artificial Intelligence in Medicine, 134, 102435. https://doi.org/10.1016/j.artmed.2022.102435

## Latent space
<p align="center">
<img src=https://github.com/haldersourav/flow-vae/assets/42878787/ccac6ec1-c1cd-4c5b-856b-93c409551fbd alt="animated" />
</p>

## Effect of peristaltic contraction strength ($\theta_c$)
<p align="center">
<img src=https://github.com/haldersourav/flow-vae/assets/42878787/e0d073ed-3cf1-4c5d-8f16-dad9dfb776f3 alt="animated" />
</p>

## Effect of active relaxation ($\theta_r$)
<p align="center">
<img src=https://github.com/haldersourav/flow-vae/assets/42878787/a335d7e2-72ff-446e-9721-e1337c014430 alt="animated" />
</p>

## Combined effect of contraction strength and active relaxation
<p align="center">
<img src=https://github.com/haldersourav/flow-vae/assets/42878787/d2c301b7-150e-44b5-a122-3e689ed16b4b alt="animated" />
</p>

More details of the physics-based problem definition and numerical solution can be found in the following papers:
1) Acharya, S., Kou, W., Halder, S., Carlson, D. A., Kahrilas, P. J., Pandolfino, J. E., and Patankar, N. A. (March 24, 2021). "Pumping Patterns and Work Done During Peristalsis in Finite-Length Elastic Tubes." ASME. J Biomech Eng. July 2021; 143(7): 071001. https://doi.org/10.1115/1.4050284
2) Elisha, G., Acharya, S., Halder, S. et al. Peristaltic regimes in esophageal transport. Biomech Model Mechanobiol 22, 23–41 (2023). https://doi.org/10.1007/s10237-022-01625-x




