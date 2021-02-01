The code was first made to call multiple files for multiple tasks.
In this version you can find the minimal code needed for the project.
If needed, the rest can be accessed in the Artemis RIG, just ask for it.

The file 'grid_search' in the main one. It calls different files as followed to train models, test it and plot what is needed.

* data.py : import and precess the raw data
* data2.py : create the generators
* model.py : build models needed for the training.
Note 1 : many more models have been tested along the project.
Note 2 : TD_2DCNN_RNN_multi_input and ThreeDCNN_RNN_multi_input are models given and writen elswhere, ask if needed.

Writer : Adrien Triquet - adrien.triquet@icloud.com