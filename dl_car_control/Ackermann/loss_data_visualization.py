#!/usr/bin/env python
import os
import numpy as np
import matplotlib.pyplot as plt
import csv



def main():

    loss_array= []
    n_array = []
    count = 1

    path = os.getcwd()


    data_file = open(path + "/last_train_data_deepracer.csv", "r")
    reader_csv = csv.reader(data_file) 

    first_line = True
    total_time = 0
    min = 20000
    max = -1
    for line in reader_csv:
        
        if first_line:
            first_line = False
            continue
                    


        loss_array.append(float(line[1]))
        n_array.append(count)


        count = count + 1
        

    
    data_file.close()

    plt.plot(n_array, loss_array, label = "Loss value", color='b')
    plt.title("Loss evolution") 
    plt.xlabel('Epochs')
    plt.ylabel('Loss value')
    plt.legend(loc="upper left")

    plt.show()

    print("FIN")



# Execute!
if __name__ == "__main__":
    main()
