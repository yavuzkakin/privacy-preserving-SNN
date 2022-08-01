import crypten
from crypten import mpc
import torch
crypten.init()

@mpc.run_multiprocess(world_size=2)
def encrypt(tensor1, tensor2):
    ct1 = crypten.cryptensor(tensor1, src=0)
    ct2 = crypten.cryptensor(tensor2, src=1)

    ct3 = ct1*ct2
    return ct3

encrypt(torch.tensor([1]), torch.tensor([2]))



