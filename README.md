# Parameter Exploration of the Human TVB-AdEx brain-scale model
The Human TVB-AdEx brain-scale model consists of a network of mesoscopic models of neural activity, whose connection strengths are determined by diffusion weighted MRI scans that allow to track the amount of white matter spanning two different regions of the brain. The TVB-AdEx can simulate conscious-like patterns of activity when the spike-frequency adaptation (SPA) parameter is low, and unconscious-like patterns of activity when the (SPA) is higher.

However, the TVB-AdEx model is a multi-scale containing many different parameters whose impact on the dynamics had not yet been characterised. Thus, I embarked on the task of characterizing the influence of these parameters on the model by means of a thorough parameter sweep exploration.

This repository is the result of this work, containing multiple studies and scripts to run and analyze the results of the parameter sweeps on the Human TVB-AdEx Model.

## Folders
### additional_scripts
Contains scripts that I used to run simulations of the TVB-AdEx on a local machine for multiple simple studies such as understanding how the random seeding worked, plotting the TVB-AdEx behavior for certain parameter combinations, plotting the BOLD signals generated with the model + balloon model, etc. Also, to learn how to use the rank variable from MPI to parallelize code (read *ParameterSweepTutorial.pdf* for more).

### data
Contains different data that is used later on for analyses of the parameter sweep. Mainly, data that is computationally expensive to post-process, so that there is no need to manipulate it again.

### figures
The figures folder has been emptied due to the large amount of figures I had produced. The figures should be reproduced easily by running again the code. If errors of missing folders appear, just by creating them inside the figures folder should be enough.

### JUSUFlike
One of the most important directories of the repository. It is structured in a very similar way to how I structured my files on the JUSUF supercomputer. It's the main folder when following the *ParameterSweepTutorial.pdf*. Additionally, **contains the scripts that are used to analyze the results of the parameter sweeps** (mainly `processing_results.py`) in the Jupyter Notebook, **as well as the results from the parameter sweep for two connectomes** (in *./JUSUFlike/Project/FinalResults/* and *./JUSUFlike/Project/FinalResultsNew/*).

### more_connectomes
The *tvb_model_reference* directory that this folder contains uses a second connectome (the one used to obtain *./JUSUFlike/Project/FinalResultsNew/*) and we can make use of the `sims_for_plots.py` inside this folder to simulate the TVB-AdEx model with said connectome.

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
In order to connect to JUSUF (supercomputer where the parameter simulation was run) and to be able to run the parameter exploration codes stored in the folder *./JUSUFlike*, one needs to go through several steps. Usually, the PI of the project writes a document explaining the need for the computational resources. If it gets accepted, it is possible to create an account and connect via ssh to the supercomputer.

If the reader has been granted access and wants to use this repository as a guide to run a parameter exploration of the TVB-AdEx (or any other model for that case), the document *ParameterSweepTutorial.pdf* describes the steps I followed to execute a parameter sweep using MPI and Python.
The *./JUSUFlike/* folder contains the file structure that I used. The user will have a folder for their project in two partitions of the supercomputer: *Scratch* and *Project*. The first one is typically used to store temporal data files, which tend to be quite large and that are used throughout the execution of the scripts. In the latter, scripts and necessary data are stored, along with the final results. More detailed in the document.