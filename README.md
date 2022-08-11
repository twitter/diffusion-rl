# Diffusion-Offline-RL
In this work, we propose Diffusion-QL which utilizes a diffusion model as a highly expressive policy class for behavior cloning and policy regularization. In our approach we learn an action-value function and we add a term maximising action-values to the the training loss of the diffusion model, which results in a loss that seeks optimal actions that are near the behavior policy. 

## Dependencies
Plese see the ``requirements.txt`` file for the detailed python package dependencies for our project. 

## Run our Code
Running our code is quite easy, such as an example below, 
```.bash
python run_offline.py --env_name walker2d-medium-expert-v2 --algo pcq 
```
