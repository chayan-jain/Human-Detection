# Human-Detection

Human Detection uses CNN for identifying the presence of human in a image.
To save the time for training, i have used the transfer learning on VGG16 model.
I have implemented in databricks for faster execution which uses spark as backend.

Dataset used is the INRIA dataset.
I have trained the network on the subset of INRIA dataset.

Dataset link :-> http://pascal.inrialpes.fr/data/human/  (After the Negative Windows Section)

I am not able to upload the dataset because it is large in size (Approx 1GB).

Repo contains :-
1) .dbc file which can be used to run in Databricks Platform.
2) .html file which can be used to visualize input & output in best way.
3) .py file which can be used to run locally.
4) .ipynb file which can be used to run on Jupyter notebook.

In each step you have to take care about input dataset path.
