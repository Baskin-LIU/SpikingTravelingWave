import numpy as np
import matplotlib.pyplot as plt
import torch

def relu(x, threshold=0):
    return (x-threshold) * (x>threshold)

class Neurons(torch.nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.n_neuron = args['n_neuron']
        self.label = args['label']
        self.device = device     #setup group name       
        self.spiking = torch.zeros(self.n_neuron).to(device)
           
    def step(self, external):
        if self.label in external:
            self.spiking = torch.tensor(external[self.label]).type(torch.float).to(self.device)
        
    def reset(self):
        self.spiking = torch.zeros(self.n_neuron).to(self.device)
        return

    def state_dict(self):
        return {}

class LIFNeurons(Neurons):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.tau = args['tau']         #time constant
        self.V0 =args['V0']            #rest membrane potential
        self.V_reset = args['V_reset']#after spike hyperpolarization
        self.threshold = args['threshold']
        
        self.V = torch.full(size=(self.n_neuron,), fill_value=self.V0).to(self.device)
        self.spiking = torch.zeros(self.n_neuron).to(self.device) # membrane potential
        self.integrate = torch.zeros(self.n_neuron).to(self.device)  #integrated dendrite current
           
    def step(self, external):
        current=0
        if self.label in external:
            current = external[self.label].to(self.device)
        self.V += (-(self.V-self.V0)/self.tau + self.integrate + current)
        self.spiking = self.V >= self.threshold     
        self.V[self.spiking]=self.V_reset
        self.V = torch.clamp(self.V, self.V_reset)
        self.spiking = self.spiking.type(torch.float)                  
        
    def reset(self):
        self.V = torch.full(size=(self.n_neuron,), fill_value=self.V0).to(self.device) # membrane potential
        self.integrate = torch.zeros(self.n_neuron).to(self.device)
        self.spiking = torch.zeros(self.n_neuron).to(self.device)
        return
    
class LIFNeuronsWTA(Neurons):
    #Within layer lateral inhibition for Winner-take-All
    def __init__(self, args, device):
        super().__init__(args, device)
        self.tau = args['tau']         #time constant
        self.V0 =args['V0']            #rest membrane potential
        self.V_reset = args['V_reset']#after spike hyperpolarization
        self.threshold = args['threshold']
        self.inh = args['inh']
        
        self.V = torch.full(size=(self.n_neuron,), fill_value=self.V0).to(self.device)
        self.spiking = torch.zeros(self.n_neuron).to(self.device) # membrane potential
        self.integrate = torch.zeros(self.n_neuron).to(self.device)  #integrated dendrite current
           
    def step(self, external):
        current=0
        if self.label in external:
            current = external[self.label].to(self.device)
        self.V += (-(self.V-self.V0)/self.tau + self.integrate + current)
        self.spiking = self.V >= self.threshold     
        self.V[self.spiking]=self.V_reset
        self.V -= self.spiking.sum()*self.inh
        self.V = torch.clamp(self.V, self.V_reset)
        self.spiking = self.spiking.type(torch.float)                  
        
    def reset(self):
        self.V = torch.full(size=(self.n_neuron,), fill_value=self.V0).to(self.device) # membrane potential
        self.integrate = torch.zeros(self.n_neuron).to(self.device)
        self.spiking = torch.zeros(self.n_neuron).to(self.device)
        return
    
class AdaptiveLIFNeurons(Neurons):  #LIF neurons with adaptive threshold and refractory period
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
        
        self.V = torch.full(size=(self.n_neuron,), fill_value=self.V0).to(self.device)
        self.theta = torch.zeros(self.n_neuron).to(self.device)
        self.refra = torch.zeros(self.n_neuron).to(self.device)  #refractory count
        self.spiking = torch.zeros(self.n_neuron).to(self.device) # membrane potential
        self.integrate = torch.zeros(self.n_neuron).to(self.device)  #integrated dendrite current
           
    def step(self, external):
        current=self.basecurrent
        if self.label in external:
            current = external[self.label].to(self.device)
        
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
        
class Synapses(torch.nn.Module):
    def __init__(self, preNeurons:Neurons, postNeurons:LIFNeurons, Modulators, args, device):
        #connectome: Matrix of n_pre_neuron * n_post_neuron
        super().__init__()
        self.preNeurons=preNeurons
        self.postNeurons=postNeurons
        self.Modulators = Modulators       
        self.connectome = args['connectome']
        self.wmax=args['wmax'] 
        self.wmin=args['wmin']
        self.winit = args['winit']
        self.norm = args['norm']       
        self.w = torch.FloatTensor(self.connectome*self.winit + args['random_init']*np.random.normal(size=self.connectome.shape)).to(device)
        if 'LearningRule' in args:
            learningArgs = args['LearningRule']
            self.learnRule = learningArgs['initF'](preNeurons, postNeurons, self, Modulators, learningArgs, device)
        else:
            self.learnRule = None
        
        #self.dendrites = np.zeros_like(self.connectome)
        
        self.train = True
           
        
    def step(self): #learn: training step or testing step
        s_pre = self.preNeurons.spiking
        self.postNeurons.integrate += torch.matmul(s_pre, self.w)

    def learn(self, train=True):
        if train and self.learnRule is not None:           
            self.learnRule.learn()
            self.w = torch.clamp(self.w, self.wmin, self.wmax)

        self.postNeurons.integrate = 0

    def normalize(self, axis=0):
        if self.norm is not None:
            self.w = self.w/self.w.sum(axis=axis, keepdims=True)*self.norm

    def reset(self):
        if self.learnRule is not None:
            self.learnRule.reset()

    def state_dict(self):
        return {'w':self.w}
    
    def load_state_dict(self, state_dict):
        self.w = state_dict['w']
        

class LearningRule(torch.nn.Module):
    def __init__(self, preNeurons, postNeurons, synapses, Modulators, args, device):
        super().__init__()
        self.preNeurons=preNeurons
        self.postNeurons=postNeurons
        self.synapses = synapses       
        self.connectome = synapses.connectome
        self.device = device
        return
        
    def learn(self):
        return
    
    def reset(self):
        return


        
class TDSTDP(LearningRule):
    def __init__(self, preNeurons, postNeurons, synapses, Modulators, args, device):
        super().__init__(preNeurons, postNeurons, synapses, Modulators, args, device)
        self.tau_pre = args['tau_z_pre']
        self.tau_post = args['tau_z_post']
        self.lr=args['learningRate']
        self.plastic = torch.tensor(args['plastic']).to(device) #1 or in shape of connectome as plastic of part of neurons
        self.k1=args['k1']
        self.k2=args['k2']
        self.theta = args['theta']
        
        self.z_pre = torch.zeros(self.connectome.shape[0]).to(device)
        self.z_post= torch.zeros(self.connectome.shape[1]).to(device)
        
    def learn(self,):
        s_pre  = self.preNeurons.spiking
        s_post = self.postNeurons.spiking
        pre = torch.unsqueeze(s_pre,axis=1)*torch.relu(self.postNeurons.integrate-self.theta)
        prepost = torch.matmul(torch.unsqueeze(self.z_pre,1), torch.unsqueeze(s_post,0))
        postpre = torch.matmul(torch.unsqueeze(s_pre,1), torch.unsqueeze(self.z_post,0))
        
        self.synapses.w += self.plastic*self.lr*(prepost - self.k1*postpre - self.k2*pre)
        
        
        self.z_pre += s_pre - self.z_pre/self.tau_pre
        self.z_post += s_post  - self.z_post/self.tau_post        

    def reset(self, ):
        self.z_pre = torch.zeros(self.connectome.shape[0]).to(self.device)
        self.z_post= torch.zeros(self.connectome.shape[1]).to(self.device)

class DASTDP(LearningRule):
    def __init__(self, preNeurons, postNeurons, synapses, Modulators, args, device):
        super().__init__(preNeurons, postNeurons, synapses, Modulators, args, device)
        self.tau_pre = args['tau_z_pre']
        self.tau_post = args['tau_z_post']
        self.tau_elig = args['tau_elig']
        self.lr=args['learningRate']
        self.k = args['k']
        self.modulator = Modulators['DA']
        self.plastic = args['plastic'] #1 or in shape of connectome as plastic of part of neurons
        
        self.z_pre = torch.zeros(self.connectome.shape[0]).to(device)
        self.z_post= torch.zeros(self.connectome.shape[1]).to(device)
        self.z_elig= torch.zeros(self.connectome.shape).to(device)
        
               
    def learn(self, ):
        
        

        s_pre  = self.preNeurons.spiking
        s_post = self.postNeurons.spiking
    
        prepost = torch.matmul(torch.unsqueeze(self.z_pre,1), torch.unsqueeze(s_post,0))
        postpre = torch.matmul(torch.unsqueeze(s_pre,1), torch.unsqueeze(self.z_post,0))
        
        self.z_pre += s_pre - self.z_pre/self.tau_pre
        self.z_post += s_post  - self.z_post/self.tau_post

        self.synapses.w += self.lr*(self.modulator.pho*self.z_elig - self.k*(postpre+0.1*prepost))

        self.z_elig += prepost - self.z_elig/self.tau_elig


    def reset(self, ):
        self.z_pre = torch.zeros(self.connectome.shape[0]).to(self.device)
        self.z_post= torch.zeros(self.connectome.shape[1]).to(self.device)
        self.z_elig= torch.zeros(self.connectome.shape) .to(self.device)
        

class AChSTDP(LearningRule):
    def __init__(self, preNeurons, postNeurons, synapses, Modulators, args, device):
        super().__init__(preNeurons, postNeurons, synapses, Modulators, args, device)
        self.tau_pre = args['tau_z_pre']
        self.tau_post = args['tau_z_post']
        self.tau_elig = args['tau_elig']
        self.lr=args['learningRate']
        self.k = args['k']
        self.modulator = Modulators['ACh']
        self.plastic = torch.tensor(args['plastic']).to(device) #1 or in shape of connectome as plastic of part of neurons
        
        self.z_pre = torch.zeros(self.connectome.shape[0]).to(device)
        self.z_post= torch.zeros(self.connectome.shape[1]).to(device)
        #self.z_elig= np.zeros_like(self.connectome)
        
               
    def learn(self, ):
        
        #self.synapses.w += self.lr*self.modulator.pho*self.z_elig

        s_pre  = self.preNeurons.spiking
        s_post = self.postNeurons.spiking
    
        prepost = torch.matmul(torch.unsqueeze(self.z_pre,1), torch.unsqueeze(s_post,0))
        postpre = torch.matmul(torch.unsqueeze(s_pre,1), torch.unsqueeze(self.z_post,0))
        
        self.z_pre += s_pre - self.z_pre/self.tau_pre
        self.z_post += s_post  - self.z_post/self.tau_post

        self.synapses.w += self.lr*self.plastic*((self.synapses.wmax-self.synapses.w)*prepost - self.synapses.w*self.k*postpre)/self.synapses.wmax#
        #self.z_elig += prepost - self.k*postpre  - self.z_elig/self.tau_elig

    def reset(self, ):
        self.z_pre = torch.zeros(self.connectome.shape[0]).to(self.device)
        self.z_post= torch.zeros(self.connectome.shape[1]).to(self.device)
        #self.z_elig= np.zeros_like(self.connectome) 
        
class Modulator(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.name = args['name']  #setup name       
        self.pho = 0 #concentration
           
    def step(self):
        return
        
    def reset(self):
        self.pho = 0
        
class DopamineTD(Modulator):
    def __init__(self, args, NeuGroups):
        super().__init__(args)
        self.neurons = NeuGroups[args['neurons']]
        self.inhibition = torch.zeros(args['inhLag'])
        self.tau = args['tau']
        self.base = args['base']
           
    def step(self, external):
        injection=0
        if self.name in external:
            injection = external[self.name]
        s = self.neurons.spiking.sum().cpu()
        self.pho += -(self.pho-self.base)/self.tau + s - self.inhibition[-1] + injection
        
        #push the array
        self.inhibition.roll(1)
        self.inhibition[0] = s
                
        
    def reset(self):
        self.pho = 0

class AChNovel(Modulator):
    def __init__(self, args, NeuGroups):
        super().__init__(args)
        self.tau = args['tau']
           
    def step(self, external):
        injection=0
        if self.name in external:
            injection = external[self.name]
        self.pho += -self.pho/self.tau + injection
                
        
    def reset(self):
        self.pho = 0
                
    
class Network(torch.nn.Module):
    def __init__(self, NeuronsGroups, Connections, Modulation, device):
        super().__init__()

        self.NeuGroups = {}
        self.SynGroups = {}
        self.Modulators = {}
        
        #initiate neuron groups
        for groupArgs in NeuronsGroups:
            self.NeuGroups[groupArgs['label']] = groupArgs['initF'](groupArgs, device)
            
        #initiate modulators
        if Modulation is not None:
            for modulator in Modulation:
                self.Modulators[modulator['name']] = modulator['initF'](modulator, self.NeuGroups)
        
        #initiate synapses
        for connectionArgs in Connections:
            preName = connectionArgs['preNeurons']
            postName = connectionArgs['postNeurons']
            preNeurons=self.NeuGroups[preName]
            postNeurons=self.NeuGroups[postName]
            self.SynGroups[preName+'-'+postName] = Synapses(preNeurons, postNeurons, self.Modulators, connectionArgs, device)     
        


    def run(self,external):
        for _,connection in self.SynGroups.items():
            connection.step()
        for _, group in self.NeuGroups.items():
            group.step(external)
        for _, modulator in self.Modulators.items():            
            modulator.step(external)
        for _,connection in self.SynGroups.items():
            connection.learn()
        self.recorder()

    def reset(self):
        for _,connection in self.SynGroups.items():
            connection.reset()
        for _, group in self.NeuGroups.items():
            group.reset()
        for _, modulator in self.Modulators.items():
            modulator.reset()

    def normalize(self, axis=0):
        for _,connection in self.SynGroups.items():
            connection.normalize(axis)

    def recorder(self):
        return

    def save(self, path='./ckp/net.pt'):
        state_dict = {}
        for label, group in self.NeuGroups.items():
            state_dict[label] = group.state_dict()
        for label, connection in self.SynGroups.items():
            state_dict[label] = connection.state_dict()
        torch.save(state_dict, path)

    def load(self, path='./ckp/net.pt'):
        state_dict = torch.load(path)
        for label, group in self.NeuGroups.items():
            group.load_state_dict(state_dict[label])
        for label, connection in self.SynGroups.items():
            connection.load_state_dict(state_dict[label])

    def s(self, label):
        return self.NeuGroups[label].spiking
    
    def w(self, label):
        return self.SynGroups[label].w