# Training DNC

import sys
import time
import torch
import random
import numpy as np

# torch.autograd.set_detect_anomaly(True) # Setting Anomaly Detection True for finding bad operations

def random_seed():
    seed = int(time.time()*10000000)
    random.seed(seed)
    np.random.seed(int(seed/10000000))      # NumPy seed Range is 2**32 - 1 max
    torch.manual_seed(seed)

def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "1":
            from tasks.copy_task import task_copy
            c_task = task_copy()                    # Initialization of the Copy Task
            print("\nStarting Copy Task for DNC\n")
        elif sys.argv[1] == "2":
            from tasks.babi_task import task_babi
            c_task = task_babi()                    # Initialization of the bAbI Task
            print("\nStarting bAbI Question Answering Task for DNC\n")
        else:
            print("Task does not exist...")
            exit()
    else:
        print("Error: Enter the task for DNC training!\n1:Copy \n2:bAbI\n")
        exit()

    epoch = sys.argv[3] # Last Epoch number till the model was trained (eg: 0) (Not Applicable for Copy Task)
    batch = sys.argv[4] # Last Batch Number till the model was trained (eg: 1000)
    batch_size = 1

    # Random Seed
    random_seed()

    c_task.init_dnc()
    c_task.init_loss()
    c_task.batch_size = batch_size

    if c_task.get_task_name() == "copy_task" :
        c_task.load_model(2, batch)
        loss, cost, inp, labels, prediction = c_task.test_model()
    elif c_task.get_task_name() == "bAbI_task" :
        c_task.load_model(2, epoch, batch)
        accuracy = c_task.test_model()

if __name__ == '__main__':
    main()