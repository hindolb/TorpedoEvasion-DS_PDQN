# TorpedoEvasion-DS_PDQN
The repository contains files which implements the paper: "Dynamic Multi Threat Evasion in Autonomous Vehicles using Parameterized Actions". The repository consists of three python files which are as follows:
1. Environment_PDQN_DS.py: This python files implements the underwater environment and simulates the torpedo evasion events. The action of AUV is as per the command received from the RL agent (implemented in PDQN_DS.py). The torpedo follows proportional navigation using which it tries to home in on the AUV. If decoy(s) launched by the AUV is able to succesfully deceive the torpedo, the proportional navigation based homing logic targets the decoy and gets destroyed. This file is not directly executable and is a helper file. 
2. PDQN_DS.py: This python files implements the DeepSets redefined PDQN which controls the RL agent such that intelligent actions of the AUV can be generated. The actions generated from this RL agent is used in Environment_PDQN_DS.py file for simulating the actions of the AUV. This file is also not directly executable and is a helper file. 
3. Training_DS.py: This isthe executable file with the main function. The file controls the RL episodes by running the underwater environment simulation, observes and abstracts the state of the environment and calculates rewards. Since, we are using an Off-Policy RL algorithm, a replay buffer is used which is also populated. When appropriate, samples are taken from the replay buffer and the RL agent is trained.

Execution: 

1. Download the files   
2. cd to the folder at the location of the downloaded files
3. Run $python Training_DS.py
4. Console output: Results of training to monitor progress
5. File output: Trained model is saved at an interval of 100 episodes. At the end of training, the trained model is loaded which can be used for effective control of the AUV actions.
