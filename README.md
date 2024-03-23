# OCT Scan Image Denoising with Diffusion Models
Optical coherene tomography (OCT) is a non-invasive imaging technique that is used in many medical areas. However these OCT scans suffer from inherent speckle noise patterns which can negatively impact their medical use. In recent years many deep learning approaches for OCT denosing have been proposed, such as Denoising Diffusion Probabilistic Models (DDPMs). These models consist of a forward process where Gaussian noise in gradually added to the input until it is completely destroyed and a reverse process where the model tries to reconstruct the original input. Due to this reverse process, diffusion models can be adapted to denoising tasks without the need for noise-free ground-truth references during inference. In this project they are addapted to retinal OCT image denoising. The network is trained on the [OCTID dataset](https://arxiv.org/abs/1812.07056) by the University of Waterloo and tested on the [RETOUCH dataset](https://optima.meduniwien.ac.at/wp-content/uploads/2021/12/2019_HBogunovic_RETOUCH_IEEETransactions_preprint.pdf) by the Medical University of Vienna. 

![example results](images/merged.jpg "example results")

# Code structure
In this repository the source code for pre-processing as well as training the network is given in the *src* folder. Also a checkpoint of the trained model is provided.

# References
This project is based on the works of Hu et al. about *Unsupervised Denoising of Retinal OCT with Diffusion Probabilistic Model* which can be found [here](https://arxiv.org/pdf/2201.11760.pdf). It directly uses their network architecture which can be found on their [GitHub repository](https://github.com/DeweiHu/OCT_DDPM). The network was trained on the OCTID dataset which can be accessed [here](https://www.openicpsr.org/openicpsr/project/108503/version/V1/view). Testing was done on the RETOUCH dataset which can be found [here](https://retouch.grand-challenge.org/). 

This project was part of a pratical work at the Institute of Machine Learning at Johnnes Kepler University in Linz in colaboration with the Medical University of Vienna.
