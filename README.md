# timbre_analysis_synthesis

This project is based on the sms-tools which is developed by MTG. Their code can be found at https://github.com/MTG/sms-tools. The timbre modification part and the modification GUI is originate with Han Zhang. 

All the codes can be cloned from this repo, but please refer to the instructions of the sms-tools first to compile some C functions. Besides the modules they mentioned in their document, this code also need the following modules: copy, json, sklearn, joblib. Please install before use the software.

To launch the interface, please go to the directory software/modification_interface and run modification_GUI.py: 
$	python3 modification_GUI.py
Python versions higher than 3.7 are recommended.

Trained classifiers for instrument recognition are already included in the files. If you want to train it yourself, please contact the auther for the structured dataset.

Details about this project are explained here: https://zhanghanpqqo.github.io/HanZhang/research.html
Enjoy!
