from re import I
import time
import crypten
import numpy as np
import matplotlib.pyplot as plt
import logging


#logging.basicConfig(filename='example.log',level=logging.INFO)
#logging.getLogger('matplotlib').setLevel(logging.INFO)

crypten.init()


def time_measure(func):
    def wrapper():
        time_start = time.time()
        func()
        time_m = time.time()-time_start
        return time_m
    return wrapper

@crypten.mpc.run_multiprocess(2)
@time_measure
def foo():
    ct1 = crypten.cryptensor([[0.01]],src=0)
    ct2 = crypten.cryptensor([[0.01]],src=1)

    ct3 = ct1*ct2

def average_time(l):
    av=[0 for i in range(len(l[0]))]
    print(l)
    av = np.array(av, dtype=float)
    for i in l:
        av+= np.array(i)
    return av/len(l)

def run_exp(steps):
    time_l= []
    bits = [i for i in range(2,40,1)]
    for _ in range(steps):
        time_l.append([])
        for b in bits:
            crypten.config.cfg.encoder.precision_bits=b
            t = foo()
            time_l[_].append(max(t))    
        print(_)
    return time_l
    
    
res=average_time(run_exp(20))
print(res)
plt.plot([i for i in range(2,40,1)], res)
plt.show()

# standard dev
# weghts 10-2