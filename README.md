# TorpedoEvasion-DS_PDQN
The repository contains files which implements the paper: "Dynamic Multi Threat Evasion in Autonomous Vehicles using Parameterized Actions". The repository consists of three python files which are as follows:
1. Environment_PDQN_DS.py: This python files implements the underwater environment and simulates the torpedo evasion events. The action of AUV is as per the command received from the RL agent (implemented in PDQN_DS.py). The torpedo follows proportional navigation using which it tries to home in on the AUV. If decoy(s) launched by the AUV is able to succesfully deceive the torpedo, the proportional navigation based homing logic targets the decoy and gets destroyed. This file is not directly executable and is a helper file. 
2. PDQN_DS.py
3. Training_DS.py
