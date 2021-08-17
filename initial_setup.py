import yaml
from helper_setup import *

if __name__ == '__main__':
    stream = open("config.yaml", 'r')
    config = yaml.safe_load(stream)
    
    iteration = config["active_learning_iteration"]
    response = input("Next iteration number is " + str(iteration) + ". Would you like to identify samples to annotate for this iteration or build a dataset from a previous iteration using the log file. To build next iteration, input [Y]. Else, put the iteration number of interest.\n")
    
    builder = Dataset_Builder(config)
    if response == "Y" or response == "y":
        builder.build_next_iteration()
    else:
        #check that the iteration exists
        if int(response) > iteration:
            raise ValueError("Desired iteration number is higher than latest")
        builder.build_from_log(response)