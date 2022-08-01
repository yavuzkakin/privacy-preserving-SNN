from tokenize import String
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import torch
from torch import tensor
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
from examples.multiprocess_launcher import MultiProcessLauncher
import argparse
import snn
logging.basicConfig(filename='example.log',level=logging.DEBUG)

def run_middleware(args):
    snn.compute_classification_accuracy_test(
        args.x,
        args.y,
        args.w1,
        args.w2,
        args.device
    )

def main(run_experiment,args,mp=True,ws=2):
    if mp:
        launcher = MultiProcessLauncher(ws, run_experiment,args)
        launcher.start()
        launcher.join()
        launcher.terminate()
    else:
        run_experiment()

      
if __name__ == "__main__":

    device = torch.device("cuda")
    
    w1,w2 = snn.load_weigths()
    x_train, y_train, x_test, y_test = snn.data_preprocess()
    #w1.to(device)
    #w2.to(device)
        
    parser = argparse.ArgumentParser(description="CrypTen Cifar Training")
    parser.add_argument(
        "--x",
        type=np.array,
        default=x_test[100:200],
        help="First tensor, x",
    )
    parser.add_argument(
        "--y",
        type=np.array,
        default=y_test[100:200],
        help="Second tensor, y",
    )
    parser.add_argument(
        "--w1",
        type=tensor,
        default=w1,
        help="First weight, w1. Load it if pre-trained",
    )
    parser.add_argument(
        "--w2",
        type=tensor,
        default=w2,
        help="Second weight, w1. Load it if pre-trained",
    )
    parser.add_argument(
        "--device",
        type=torch.device,
        default=device,
        help="Second tensor, y",
    )

    main(run_middleware,args = parser.parse_args())

