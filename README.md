# Parameter Exploration of the Human TVB-AdEx brain-scale model
The Human TVB-AdEx brain-scale model consists of a network of mesoscopic models of neural activity, whose connection strengths are determined by diffusion weighted MRI scans that allow to track the amount of white matter spanning two different regions of the brain. The TVB-AdEx can simulate conscious-like patterns of activity when the spike-frequency adaptation (SPA) parameter is low, and unconscious-like patterns of activity when the (SPA) is higher.

However, the TVB-AdEx model is a multi-scale containing many different parameters whose impact on the dynamics had not yet been characterised. Thus, I embarked on the task of characterizing the influence of these parameters on the model by means of a thorough parameter sweep exploration.

This repository is the result of this work, containing multiple studies and scripts to run and analyze the results of the parameter sweeps on the Human TVB-AdEx Model.

## Remaining work to do:
- Generate the tutorial.pdf doc
- Finish this readme
- Download and re-organize the JUSUF scripts (will need to do it at home this weekend)

## Folders
### additional_scripts

### data

### figures
The figures folder has been emptied due to the large amount of figures I had produced. The figures should be reproduced easily by running again the code. If errors of missing folders appear, just by creating them inside the figures folder should be enough.

### JUSUFlike

### more_connectomes

### tvb_model_reference
Contains the standard library used to run a simulation of the human TVB-AdEx model. 

## Jupyter Notebooks
The names of the jupyter notebooks themselves are already quite self-descriptive. If, with the help of the commented code inside, there are still doubts of what is going on inside a library, don't hesitate to contact me.

## Technical Requirements
### Local Machine
**Python version** = 3.8.10
Used Windows 10 to generate the requirements.txt. Although most of the project was developed in an ubuntu machine.

It would be best practice to set up a virtual environment, activate it, and then install the libraries in `./requirements.txt`. This can be easily done by running the following line of code on the terminal, with the virtual environment active:
`pip install -r requirements.txt`

### JUSUF
In order to connect to JUSUF (supercomputer where the parameter simulation was run) and to be able to run the parameter exploration codes stored in the folder `./JUSUFlike`, one needs to go through several steps. Usually, the PI of the project writes a document explaining the need for the computational resources. If it gets accepted, it is possible to create an account and connect via ssh to the supercomputer.

If the reader has been granted access and wants to use this repository as a guide to run a parameter exploration of the TVB-AdEx (or any other model for that case), the document `./JUSUFlike/tutorial.pdf` describes the steps I followed to execute a parameter sweep using MPI and Python.
The `./JUSUFlike/` folder contains the file structure that I used. The user will have a folder for their project in two partitions of the supercomputer: `SCRATCH` and `PROJECT`. The first one is typically used to store temporal data files, which tend to be quite large and that are used throughout the execution of the scripts. In the latter, scripts and necessary data are stored, along with the final results. More detailed in the document.