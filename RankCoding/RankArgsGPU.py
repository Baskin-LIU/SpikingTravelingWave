import sys
sys.path.insert(0, 'C:/Users/Bingkun LIU/Desktop/Ongoing Rearch/Spiking')

from SpikingNetCuda import *

mapsize=(60, 60)
groupsize=1
inputsize = 464#-134

def MexicoHat2Ddonut(mapsize, groupsize, var_p, var_n, p, n, base):
    w = np.zeros((mapsize[0]*mapsize[1]*groupsize, mapsize[0]*mapsize[1]*groupsize))
    for i in range(mapsize[0]):
        for j in range(mapsize[1]):
            pos = w[(i*mapsize[1]+j)*groupsize:(i*mapsize[1]+j+1)*groupsize]
            for k in range(mapsize[0]):
                for z in range(mapsize[1]):
                    disSquare = min((i-k)**2, (i-k+mapsize[0])**2, (i-k-mapsize[0])**2) + min((j-z)**2, (j-z+mapsize[1])**2, (j-z-mapsize[1])**2)
                    if disSquare/var_n > 4 or disSquare==0:
                        pos[:, (k*mapsize[1]+z)*groupsize:(k*mapsize[1]+z+1)*groupsize] = base
                    else:
                        pos[:, (k*mapsize[1]+z)*groupsize:(k*mapsize[1]+z+1)*groupsize] = p*np.exp(-disSquare/var_p) - n*np.exp(-disSquare/var_n) + base
                
    return w

def MexicoHat2Dball(mapsize, groupsize, var_p, var_n, p, n, base):
    w = np.zeros((mapsize[0]*mapsize[1]*groupsize, mapsize[0]*mapsize[1]*groupsize))
    for i in range(mapsize[0]):
        for j in range(mapsize[1]):
            pos = w[(i*mapsize[1]+j)*groupsize:(i*mapsize[1]+j+1)*groupsize]
            for k in range(mapsize[0]):
                for z in range(mapsize[1]):
                    disSquare = min((i-k)**2+(j-z)**2, (i-z+mapsize[1])**2+(j-k+mapsize[0])**2, (j-k+mapsize[0])**2+(i-z+mapsize[1])**2)
                    if disSquare/var_n > 4 or disSquare==0:
                        pos[:, (k*mapsize[1]+z)*groupsize:(k*mapsize[1]+z+1)*groupsize] = base
                    else:
                        pos[:, (k*mapsize[1]+z)*groupsize:(k*mapsize[1]+z+1)*groupsize] = p*np.exp(-disSquare/var_p) - n*np.exp(-disSquare/var_n) + base
                
    return w

default_Neurons_args={
        'tau': 20.0,
        'V_reset': -70.0,
        'V0':-65.0,
        'threshold':-45.0,
    }

default_Synapses_args={
        'preNeurons': 'neurons',
        'postNeurons': 'neurons',
        'wmax' : None,
        'wmin' : None,
        'random_init':0,
        'norm':None,
    }

NeuronsGroups = [
    {
        **default_Neurons_args,
        'label': 'input',
        'n_neuron': inputsize,
        'initF': Neurons
    },
    {
        **default_Neurons_args,
        'label': 'fluid',
        'n_neuron': mapsize[0]*mapsize[1]*groupsize,
        'initF': AdaptiveLIFNeurons,
        'V_reset': -70.0,
        'tau_theta' : 1e6,  # tau for adaptive threshold
        'k_theta': 1e-4,  # step_size for adaptive threshold
        'refractory': 10,
        'base_current': 1.4,
    },
    ]

Connections = [
    {
        **default_Synapses_args,
        'preNeurons':'input',
        'postNeurons':'fluid',
        'connectome': np.ones((inputsize, mapsize[0]*mapsize[1]*groupsize)),
        'wmin':0.0,
        'wmax':0.1,
        'winit':0.005,
        'random_init':1e-3,
        'norm':1,
        'LearningRule' : {
            'initF':AChSTDP,
            'plastic':1,
            'tau_z_pre' : 15.0,
            'tau_z_post' : 20.0,
            'tau_elig':50.0,
            'learningRate':1e-3,
            'k' : 0.6,
        }
    },
    {
        **default_Synapses_args,
        'preNeurons':'fluid',
        'postNeurons':'fluid',
        'connectome': MexicoHat2Ddonut(mapsize, groupsize, var_p=2, var_n=8, p=1.6, n=0.4, base=-0.2), #n=0.3
        'winit':25.,
    }
]

Modulation = [
    {
        'name': 'ACh',
        'initF': AChNovel,
        'inhLag': 40, #int
        'tau':40.0
        
    }
]