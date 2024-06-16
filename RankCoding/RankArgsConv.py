import sys
sys.path.insert(0, 'C:/Users/Bingkun LIU/Desktop/Ongoing Rearch/Spiking')

from SpikingNetCuda import *

mapsize=(40, 40)
inputsize = 464#-134

class AdaptiveLIFNeuronsEIConv(Neurons):  #LIF neurons with adaptive threshold and refractory period
    def __init__(self, args, device):
        super().__init__(args, device)
        self.tau = args['tau']         #time constant
        self.V0 =args['V0']            #rest membrane potential
        self.V_reset = args['V_reset']#after spike hyperpolarization
        self.threshold = args['threshold']
        self.tau_theta = args['tau_theta']
        self.k_theta = args['k_theta']
        self.refractory_period = args['refractory']
        self.basecurrent = args['base_current']
        self.E = torch.tensor(args['E']).to(self.device)
        self.I = torch.tensor(args['I']).to(self.device)
        self.global_inh = args['global_inh']
        
        self.V = torch.full(size=(self.n_neuron,), fill_value=self.V0).to(self.device)
        self.theta = torch.zeros(self.n_neuron).to(self.device)
        self.refra = torch.zeros(self.n_neuron).to(self.device)  #refractory count
        self.spiking = torch.zeros(self.n_neuron).to(self.device) # membrane potential
        self.integrate = torch.zeros(self.n_neuron).to(self.device)  #integrated dendrite current
        
        self.MexiConv = torch.nn.Conv2d(2, 1, args["ker_size"], stride=1, padding='same', padding_mode='circular', bias=False)
        with torch.no_grad():
            self.MexiConv.weight.data = torch.tensor(np.stack([args["w_e"], args['w_i']], axis=0)).unsqueeze(0).to(self.device)

        #print(self.MexiConv.weight.data)
           
    def step(self, external):
        current=self.basecurrent
        if self.label in external:
            current = external[self.label].to(self.device)
        
       
        EI = torch.stack((self.spiking.view(mapsize)*self.E, self.spiking.view(mapsize)*self.I), dim=0)
        self.integrate += self.MexiConv(EI.unsqueeze(0)).view(-1)
        self.integrate -= self.global_inh*(EI[1].sum()) #global inhibition
        
        #print(self.integrate)

        self.V += -(self.V-self.V0)/self.tau + (self.refra<=0)*(self.integrate + current)
        
        self.spiking = self.V >= (self.theta+self.threshold) 
        #print(self.V)    
        self.V[self.spiking]=self.V_reset

        self.refra -= 1
        self.refra[self.spiking] = self.refractory_period

        self.spiking = self.spiking.type(torch.float)
        self.theta += -self.theta/self.tau_theta + self.k_theta*self.spiking
        self.V = torch.clamp(self.V, self.V_reset)
        
        
    def reset(self):
        self.V = torch.full(size=(self.n_neuron,), fill_value=self.V0).to(self.device) # membrane potential
        self.integrate = torch.zeros(self.n_neuron).to(self.device)
        self.spiking = torch.zeros(self.n_neuron).to(self.device)
        self.refra = torch.zeros(self.n_neuron).to(self.device)
    
    def state_dict(self):
        return {'theta':self.theta}
    
    def load_state_dict(self, state_dict):
        self.theta = state_dict['theta']



def MexicoHat2DEI(mapsize, var_p, var_n, p, n, ker_size, scale):
    w_e = np.zeros((ker_size, ker_size))
    w_i = np.zeros((ker_size, ker_size))
    center = ker_size//2
    for i in range(ker_size):
        for j in range(ker_size):
            if i==center and j==center:
                continue
            disSquare = (i-center)**2+(j-center)**2
            w_e[i,j] = scale*p*np.exp(-disSquare/var_p)
            w_i[i,j] = scale*n*np.exp(-disSquare/var_n)  
    global_inh=5*scale*w_i[0,0]
    w_i-=global_inh

    E = np.ones(mapsize)
    I = np.zeros(mapsize)
    for i in range(mapsize[0]):
        for j in range(mapsize[1]):
            if i%2==0 and j%2==0:
                I[i,j]=1
    E-=I           
    return E, I, w_e, w_i, ker_size, global_inh





E, I, w_e, w_i, ker_size, global_inh = MexicoHat2DEI(mapsize, var_p=1.5, var_n=10., p=0.8, n=0.2, ker_size=9, scale=5)
print(global_inh)

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
        'n_neuron': mapsize[0]*mapsize[1],
        'initF': AdaptiveLIFNeuronsEIConv,
        'V_reset': -70.0,
        'tau_theta' : 1e6,  # tau for adaptive threshold
        'k_theta': 4e-4,  # step_size for adaptive threshold
        'refractory': 8,
        'base_current': 1.2,
        'E': E,
        'I': I,
        'w_e': w_e,
        'w_i': -w_i,
        'global_inh': global_inh,
        'ker_size': ker_size,
    },
    ]

Connections = [
    {
        **default_Synapses_args,
        'preNeurons':'input',
        'postNeurons':'fluid',
        'connectome': np.ones((inputsize, mapsize[0]*mapsize[1])),
        'wmin':0.0,
        'wmax':0.1,
        'winit':0.005,
        'random_init':1e-3,
        'norm':0.2,
        'LearningRule' : {
            'initF':AChSTDP,
            'plastic':1.,
            'tau_z_pre' : 15.0,
            'tau_z_post' : 20.0,
            'tau_elig':50.0,
            'learningRate':1e-3,
            'k' : 0.6,
        }
    },

]

Modulation = [
    {
        'name': 'ACh',
        'initF': AChNovel,
        'inhLag': 40, #int
        'tau':40.0
        
    }
]