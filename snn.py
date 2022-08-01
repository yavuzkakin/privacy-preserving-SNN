import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import torch
import torch.nn as nn
import torchvision
import pandas as pd
import crypten

# The coarse network structure is dicated by the Fashion MNIST dataset. 
nb_inputs  = 28*28
nb_hidden  = 100
nb_outputs = 10
time_step = 1e-3
nb_steps  = 100
batch_size = 256
dtype = torch.float

tau_mem = 10e-3
tau_syn = 5e-3
alpha   = float(np.exp(-time_step/tau_syn))
beta    = float(np.exp(-time_step/tau_mem))


def data_preprocess(subset=-1):
    # Here we load the Dataset
    root = os.path.expanduser("mnist")
    train_dataset = torchvision.datasets.FashionMNIST(root, train=True, transform=None, target_transform=None, download=True)
    test_dataset = torchvision.datasets.FashionMNIST(root, train=False, transform=None, target_transform=None, download=True)

    x_train = np.array(train_dataset.data, dtype=np.float)
    x_train = x_train.reshape(x_train.shape[0],-1)/255
    x_test = np.array(test_dataset.data, dtype=np.float)
    print(x_test.shape)
    x_test = x_test.reshape(x_test.shape[0],-1)/255
    print(x_test[90])    

    y_train = np.array(train_dataset.targets, dtype=np.int)
    y_test  = np.array(test_dataset.targets, dtype=np.int)
    print(y_test[:90])    
    return x_train, y_train, x_test[:subset],y_test[:subset]

def current2firing_time(x, tau=20, thr=0.2, tmax=1.0, epsilon=1e-7):
    """ Computes first firing time latency for a current input x assuming the charge time of a current based LIF neuron.

    Args:
    x -- The "current" values

    Keyword args:
    tau -- The membrane time constant of the LIF neuron to be charged
    thr -- The firing threshold value 
    tmax -- The maximum time returned 
    epsilon -- A generic (small) epsilon > 0

    Returns:
    Time to first spike for each "current" x
    """
    idx = x<thr
    x = np.clip(x,thr+epsilon,1e9)
    T = tau*np.log(x/(x-thr))
    # T in [1.5,109]
    T[idx] = tmax

    return T
 

def sparse_data_generator(X, y, batch_size, nb_steps, nb_units, shuffle=True, cpu = False ):
    """ This generator takes datasets in analog format and generates spiking network input as sparse tensors. 

    Args:
        X: The data ( sample x event x 2 ) the last dim holds (time,neuron) tuples
        y: The labels
        nb_steps: number of time steps
        nb_units: 784 (28x28)
    """


    labels_ = np.array(y,dtype=np.int)
    number_of_batches = len(X)//batch_size
    sample_index = np.arange(len(X))

    # compute discrete firing times
    tau_eff = 20e-3/time_step
    firing_times = np.array(current2firing_time(X, tau=tau_eff, tmax=nb_steps), dtype=np.int)
    # [0...783]
    unit_numbers = np.arange(nb_units)

    if shuffle:
        np.random.shuffle(sample_index)

    total_batch_count = 0
    counter = 0
    while counter<number_of_batches:
        # index of the entry in the current batch
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]

        coo = [ [] for i in range(3) ]
        # batch index, absolute index in X
        for bc,idx in enumerate(batch_index):
            c = firing_times[idx]<nb_steps
            # keeps the entries such that time<100 
            times, units = firing_times[idx][c], unit_numbers[c]

            batch = [bc for _ in range(len(times))]
            coo[0].extend(batch)
            coo[1].extend(times)
            coo[2].extend(units)
        
        if cpu:
            i = torch.LongTensor(coo)
            v = torch.FloatTensor(np.ones(len(coo[0])))
        
            X_batch = torch.sparse.FloatTensor(i, v, torch.Size([batch_size,nb_steps,nb_units]))
            y_batch = torch.tensor(labels_[batch_index])
            yield X_batch, y_batch
        else:
            i = torch.LongTensor(coo).to(device)
            v = torch.FloatTensor(np.ones(len(coo[0]))).to(device)
        
            X_batch = torch.sparse.FloatTensor(i, v, torch.Size([batch_size,nb_steps,nb_units])).to(device)
            y_batch = torch.tensor(labels_[batch_index],device=device)
            yield X_batch.to(device=device), y_batch.to(device=device)
        

        counter += 1

def set_weight():
    weight_scale = 7*(1.0-beta) # this should give us some spikes to begin with

    w1 = torch.empty((nb_inputs, nb_hidden),  device=device, dtype=dtype, requires_grad=True)
    torch.nn.init.normal_(w1, mean=0.0, std=weight_scale/np.sqrt(nb_inputs))

    w2 = torch.empty((nb_hidden, nb_outputs), device=device, dtype=dtype, requires_grad=True)
    torch.nn.init.normal_(w2, mean=0.0, std=weight_scale/np.sqrt(nb_hidden))

    return w1,w2

def plot_voltage_traces(mem, spk=None, dim=(3,5), spike_height=5):
    gs=GridSpec(*dim)
    if spk is not None:
        dat = 1.0*mem
        dat[spk>0.0] = spike_height
        dat = dat.detach().cpu().numpy()
    else:
        dat = mem.detach().cpu().numpy()
    for i in range(np.prod(dim)):
        if i==0: a0=ax=plt.subplot(gs[i])
        else: ax=plt.subplot(gs[i],sharey=a0)
        ax.plot(dat[i])
        ax.axis("off")

class SurrGradSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements 
    the surrogate gradient. By subclassing torch.autograd.Function, 
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid 
    as this was done in Zenke & Ganguli (2018).
    """
    
    scale = 100.0 # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which 
        we need to later backpropagate our error signals. To achieve this we use the 
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the 
        surrogate gradient of the loss with respect to the input. 
        Here we use the normalized negative part of a fast sigmoid 
        as this was done in Zenke & Ganguli (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input/(SurrGradSpike.scale*torch.abs(input)+1.0)**2
        return grad
    
# here we overwrite our naive spike function by the "SurrGradSpike" nonlinearity which implements a surrogate gradient
spike_fn  = SurrGradSpike.apply

def run_snn(inputs,w1,w2):
    h1 = torch.einsum("abc,cd->abd", (inputs, w1))
    syn = torch.zeros((batch_size,nb_hidden), device=device, dtype=dtype)
    mem = torch.zeros((batch_size,nb_hidden), device=device, dtype=dtype)

    mem_rec = []
    spk_rec = []

    # Compute hidden layer activity
    for t in range(nb_steps):
        mthr = mem-1.0
        out = spike_fn(mthr)
        rst = out.detach() # We do not want to backprop through the reset

        new_syn = alpha*syn +h1[:,t]
        new_mem = (beta*mem +syn)*(1.0-rst)

        mem_rec.append(mem)
        spk_rec.append(out)
        
        mem = new_mem
        syn = new_syn

    mem_rec = torch.stack(mem_rec,dim=1)
    spk_rec = torch.stack(spk_rec,dim=1)

    # Readout layer
    h2= torch.einsum("abc,cd->abd", (spk_rec, w2))
    flt = torch.zeros((batch_size,nb_outputs), device=device, dtype=dtype)
    out = torch.zeros((batch_size,nb_outputs), device=device, dtype=dtype)
    out_rec = [out]
    for t in range(nb_steps):
        new_flt = alpha*flt +h2[:,t]
        new_out = beta*out +flt

        flt = new_flt
        out = new_out

        out_rec.append(out)

    out_rec = torch.stack(out_rec,dim=1)
    other_recs = [mem_rec, spk_rec]
    return out_rec, other_recs

def train(x_data, y_data, w1,w2, lr=2e-3, nb_epochs=40, save=False):
    params = [w1,w2]
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9,0.999))

    log_softmax_fn = nn.LogSoftmax(dim=1)
    loss_fn = nn.NLLLoss()
    
    loss_hist = []
    for e in range(nb_epochs):
        local_loss = []
        for x_local, y_local in sparse_data_generator(x_data, y_data, batch_size, nb_steps, nb_inputs):
            output,_ = run_snn(x_local.to_dense(),w1,w2)
            m,_=torch.max(output,1)
            log_p_y = log_softmax_fn(m)
            loss_val = loss_fn(log_p_y, y_local)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            local_loss.append(loss_val.item())
        mean_loss = np.mean(local_loss)
        print("Epoch %i: loss=%.5f"%(e+1,mean_loss))
        loss_hist.append(mean_loss)
        if save:
            save_weights(w1,w2)    
    return loss_hist

def compute_classification_accuracy(x_data, y_data,w1,w2):
    """ Computes classification accuracy on supplied data in batches. """
    accs = []
    i=0
    for x_local, y_local in sparse_data_generator(x_data, y_data, batch_size, nb_steps, nb_inputs, shuffle=False):
        output,_ = run_snn(x_local.to_dense(),w1,w2)
        m,_= torch.max(output,1) # max over time
        _,am=torch.max(m,1)      # argmax over output units
        tmp = np.mean((y_local==am).detach().cpu().numpy()) # compare to labels
        accs.append(tmp)
        i+=1
        if i==10:
          return np.mean(accs)
    return np.mean(accs)

def save_weights(w1,w2):
    torch.save(w1, 'snn_weights/w1')
    torch.save(w2, 'snn_weights/w2')
    
def load_weigths(cpu=True):
    if cpu:
        w1 = torch.load('snn_weights/w1', map_location=lambda storage, loc: storage)
        w2 = torch.load('snn_weights/w2', map_location=lambda storage, loc: storage)
    else:
        w1 = torch.load('snn_weights/w1')
        w2 = torch.load('snn_weights/w2')
        
    return w1,w2

def plot_loss(loss_hist):
    plt.figure(figsize=(3.3,2),dpi=150)
    plt.plot(loss_hist)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    sns.despine()

def spike_fn_test(x,device):
    sub=torch.zeros(x.size(),dtype=float)
    sub_enc=crypten.cryptensor(sub)
    sub_enc.to(device)
    out=x>sub_enc 
    return out
    
def run_snn_test(inputs,w1_enc,w2_enc,device):
    #h1 = torch.einsum("abc,cd->abd", (inputs, w1))
    syn = torch.zeros((batch_size,nb_hidden),  dtype=dtype)
    mem = torch.zeros((batch_size,nb_hidden),  dtype=dtype)

    syn_enc=crypten.cryptensor(syn)
    mem_enc=crypten.cryptensor(mem)
    syn_enc.to(device)
    mem_enc.to(device)

    mem_rec = []
    spk_rec = []

    # Compute hidden layer activity
    for t in range(nb_steps):
        #print("step number :"+str(t))
        mthr = mem_enc-crypten.cryptensor(1.0)
        mthr.to(device)
        out = spike_fn_test(mthr,device)
        rst = out.detach() # We do not want to backprop through the reset
        try:
            new_syn = alpha*syn_enc*alpha +inputs.matmul(w1_enc)[:,t]
            new_mem = (beta*mem_enc +syn_enc)*(1.0-rst)
            mem_rec.append(mem_enc)
            spk_rec.append(out)
            
            mem_enc = new_mem
            syn_enc = new_syn
        except Exception as e:
            print(e)



    mem_rec = crypten.stack(mem_rec,dim=1)
    spk_rec = crypten.stack(spk_rec,dim=1)

    # Readout layer
    #h2= torch.einsum("abc,cd->abd", (spk_rec, w2))
    
    
    flt = torch.zeros((batch_size,nb_outputs), device=device, dtype=dtype)
    out = torch.zeros((batch_size,nb_outputs), device=device, dtype=dtype)

    flt_enc=crypten.cryptensor(flt)
    out_enc=crypten.cryptensor(out)


    out_rec = []
    out_rec.append(out_enc)

    for t in range(nb_steps):
        #print("step number :"+str(t))
        #if t<100 and t>95:
        new_flt = alpha*flt_enc +spk_rec.matmul(w2_enc)[:,t]
        new_out = beta*out_enc +flt_enc

        flt_enc = new_flt
        out_enc = new_out

        out_rec.append(out_enc)

    out_rec = crypten.stack(out_rec,dim=1)
    other_recs = [mem_rec, spk_rec]
    return out_rec, other_recs

def compute_classification_accuracy_test(x_data, y_data,w1,w2,device):
    """ Computes classification accuracy on supplied data in batches. """

    w1_enc=crypten.cryptensor(w1,src=0)
    w2_enc=crypten.cryptensor(w2,src=0)
    w1_enc.to(device)
    w2_enc.to(device)
    
    accs = []
    i=0
    for x_local, y_local in sparse_data_generator(x_data, y_data, batch_size, nb_steps, nb_inputs, shuffle=False,cpu=True):
        x_data_enc=crypten.cryptensor(x_local.to_dense(), src=1)
        x_data_enc.to(device)
        output_enc,_ = run_snn_test(x_data_enc,w1_enc,w2_enc,device)
        output=output_enc.get_plain_text()
        m,_= torch.max(output,1) # max over time
        _,am=torch.max(m,1)      # argmax over output units
        tmp = np.mean((y_local==am).detach().cpu().numpy()) # compare to labels
        accs.append(tmp)
        print(tmp)
        i+=1
        if i==1:
          return np.mean(accs)    
    return np.mean(accs)

def test(x,y,w1,w2,device):
    x=crypten.cryptensor(x,src=0)
    x.to(device)
    print(x.shape)
    w1=crypten.cryptensor(w1,src=1)
    w1.to(device)
    print(w1.shape)
    print(x.matmul(w1))


if __name__ == "__main__":
    
    # Check whether a GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("cuda")     
    else:
        device = torch.device("cpu")
    

    w1,w2=set_weight()
    x_train, y_train, x_test, y_test = data_preprocess(subset=200)
    acc = train(x_train,y_train, w1,w2, save=True, nb_epochs=5)     

    print("Training accuracy: %.3f"%(compute_classification_accuracy(x_train,y_train,w1,w2)))
    print("Test accuracy: %.3f"%(compute_classification_accuracy(x_test,y_test,w1,w2)))
    plot_loss(acc)