
## How to run an experiment

Setup the environment by installing the requirements.txt file

Straightforwardly, we provide the main routine used to run experiments in the "<i>main.py</i>"file. We organised the file to enable the reviewer to quickly switch the experiments scenarios by only setting these three variables: 

1- ( Non-IID and Number of Attackers): Non-IID={0.5 or 0.9 CIFAR10}, IID=True specifically, from this location at code -> Experiment/CustomConfig.py  

2- Number of Attackers {just go to -> Experiment/CustomConfig.py and select the number of attacks scenario).

3- To run our methods SCRFA.py.
From this location at code -> Experiment/AggregatorConfig.py  

After running an experiment, the results will be available on test
