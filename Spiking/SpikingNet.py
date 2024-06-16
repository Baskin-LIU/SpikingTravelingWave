import numpy as np
import matplotlib.pyplot as plt

def relu(x, threshold=0):
    return (x-threshold) * (x>threshold)

class Neurons():
    def __init__(self, args):
        self.n_neuron = args['n_neuron']
        self.label = args['label']     #setup group name       
        self.spiking = np.zeros(self.n_neuron)
           
    def step(self, external):
        if self.label in external:
            self.spiking = external[self.label]
        
    def reset(self):
        self.spiking = np.zeros(self.n_neuron)
        return

class LIFNeurons(Neurons):
    def __init__(self, args):
        super().__init__(args)
        self.tau = args['tau']         #time constant
        self.V0 =args['V0']            #rest membrane potential
        self.V_reset = args['V_reset']#after spike hyperpolarization
        self.threshold = args['threshold']
        
        self.V = np.full(self.n_neuron, self.V0)
        self.spiking = np.zeros(self.n_neuron) # membrane potential
        self.integrate = np.zeros(self.n_neuron)  #integrated dendrite current
           
    def step(self, external):
        current=0
        if self.label in external:
            current = external[self.label]
        self.V += (-(self.V-self.V0)/self.tau + self.integrate + current)
        self.spiking = self.V >= self.threshold     
        self.V[self.spiking]=self.V_reset                  
        
    def reset(self):
        self.V = np.full(self.n_neuron, self.V0) # membrane potential
        self.integrate = np.zeros(self.n_neuron)
        self.spiking = np.zeros(self.n_neuron)
        return
        
class Synapses():
    def __init__(self, preNeurons:Neurons, postNeurons:LIFNeurons, Modulators, args):
        #connectome: Matrix of n_pre_neuron * n_post_neuron
        
        self.preNeurons=preNeurons
        self.postNeurons=postNeurons
        self.Modulators = Modulators       
        self.connectome = args['connectome']
        self.wmax=args['wmax'] 
        self.wmin=args['wmin']
        self.winit = args['winit']       
        self.w = self.connectome*self.winit + args['random_init']*np.random.normal(size=self.connectome.shape)
        if 'LearningRule' in args:
            learningArgs = args['LearningRule']
            self.learnRule = learningArgs['initF'](preNeurons, postNeurons, self, Modulators, learningArgs)
        else:
            self.learnRule = None
        
        #self.dendrites = np.zeros_like(self.connectome)
        
        self.train = True
           
        
    def step(self): #learn: training step or testing step
        s_pre = self.preNeurons.spiking
        self.postNeurons.integrate += s_pre.dot(self.w)

    def learn(self, train=True):
        if train and self.learnRule is not None:           
            self.learnRule.learn()
            if self.wmin:
                self.w[self.w<self.wmin]=self.wmin
            if self.wmax:
                self.w[self.w>self.wmax]=self.wmax

        self.postNeurons.integrate = 0

    def normalize(self):
        if self.learnRule is not None:
            self.w = self.w/self.w.mean(axis=1, keepdims=True)*self.winit

    def reset(self):
        self.learnRule.reset()
        

class LearningRule():
    def __init__(self, preNeurons, postNeurons, synapses, Modulators, args):
        self.preNeurons=preNeurons
        self.postNeurons=postNeurons
        self.synapses = synapses       
        self.connectome = synapses.connectome
        return
        
    def learn(self):
        return
    
    def reset(self):
        return


        
class TDSTDP(LearningRule):
    def __init__(self, preNeurons, postNeurons, synapses, Modulators, args):
        super().__init__(preNeurons, postNeurons, synapses, Modulators, args)
        self.tau_pre = args['tau_z_pre']
        self.tau_post = args['tau_z_post']
        self.lr=args['learningRate']
        self.plastic = args['plastic'] #1 or in shape of connectome as plastic of part of neurons
        self.k1=args['k1']
        self.k2=args['k2']
        self.theta = args['theta']
        
        self.z_pre = np.zeros_like(self.connectome[:,0])
        self.z_post= np.zeros_like(self.connectome[0,:])
        
    def learn(self,):
        s_pre  = self.preNeurons.spiking
        s_post = self.postNeurons.spiking
        pre = np.expand_dims(s_pre,axis=1)*relu(self.postNeurons.integrate, self.theta)
        prepost = np.expand_dims(self.z_pre,axis=1).dot(np.expand_dims(s_post,axis=0))
        postpre = np.expand_dims(s_pre,axis=1).dot(np.expand_dims(self.z_post,axis=0))
        
        self.synapses.w += self.plastic*self.lr*(prepost - self.k1*postpre - self.k2*pre)
        
        
        self.z_pre += s_pre - self.z_pre/self.tau_pre
        self.z_post += s_post  - self.z_post/self.tau_post        

    def reset(self, ):
        self.z_pre = np.zeros_like(self.connectome[:,0])
        self.z_post= np.zeros_like(self.connectome[0,:])

class DASTDP(LearningRule):
    def __init__(self, preNeurons, postNeurons, synapses, Modulators, args):
        super().__init__(preNeurons, postNeurons, synapses, Modulators, args)
        self.tau_pre = args['tau_z_pre']
        self.tau_post = args['tau_z_post']
        self.tau_elig = args['tau_elig']
        self.lr=args['learningRate']
        self.k = args['k']
        self.modulator = Modulators['DA']
        self.plastic = args['plastic'] #1 or in shape of connectome as plastic of part of neurons
        
        self.z_pre = np.zeros_like(self.connectome[:,0])
        self.z_post= np.zeros_like(self.connectome[0,:])
        self.z_elig= np.zeros_like(self.connectome)
        
               
    def learn(self, ):
        
        self.synapses.w += self.lr*self.modulator.pho*self.z_elig

        s_pre  = self.preNeurons.spiking
        s_post = self.postNeurons.spiking
    
        prepost = np.expand_dims(self.z_pre,axis=1).dot(np.expand_dims(s_post,axis=0))
        postpre = np.expand_dims(s_pre,axis=1).dot(np.expand_dims(self.z_post,axis=0))
        
        self.z_pre += s_pre - self.z_pre/self.tau_pre
        self.z_post += s_post  - self.z_post/self.tau_post
        self.z_elig += prepost - self.k*postpre  - self.z_elig/self.tau_elig

    def reset(self, ):
        self.z_pre = np.zeros_like(self.connectome[:,0])
        self.z_post= np.zeros_like(self.connectome[0,:])
        self.z_elig= np.zeros_like(self.connectome) 
        
class Modulator():
    def __init__(self, args):
        self.name = args['name']  #setup name       
        self.pho = 0 #concentration
           
    def step(self):
        return
        
    def reset(self):
        self.pho = 0

class AChSTDP(LearningRule):
    def __init__(self, preNeurons, postNeurons, synapses, Modulators, args):
        super().__init__(preNeurons, postNeurons, synapses, Modulators, args)
        self.tau_pre = args['tau_z_pre']
        self.tau_post = args['tau_z_post']
        self.tau_elig = args['tau_elig']
        self.lr=args['learningRate']
        self.k = args['k']
        self.modulator = Modulators['ACh']
        self.plastic = args['plastic'] #1 or in shape of connectome as plastic of part of neurons
        
        self.z_pre = np.zeros_like(self.connectome[:,0])
        self.z_post= np.zeros_like(self.connectome[0,:])
        #self.z_elig= np.zeros_like(self.connectome)
        
               
    def learn(self, ):
        
        #self.synapses.w += self.lr*self.modulator.pho*self.z_elig

        s_pre  = self.preNeurons.spiking
        s_post = self.postNeurons.spiking
    
        prepost = np.expand_dims(self.z_pre,axis=1).dot(np.expand_dims(s_post,axis=0))
        postpre = np.expand_dims(s_pre,axis=1).dot(np.expand_dims(self.z_post,axis=0))
        
        self.z_pre += s_pre - self.z_pre/self.tau_pre
        self.z_post += s_post  - self.z_post/self.tau_post

        self.synapses.w += self.lr*(prepost - self.k*postpre)
        #self.z_elig += prepost - self.k*postpre  - self.z_elig/self.tau_elig

    def reset(self, ):
        self.z_pre = np.zeros_like(self.connectome[:,0])
        self.z_post= np.zeros_like(self.connectome[0,:])
        #self.z_elig= np.zeros_like(self.connectome) 
        
class Modulator():
    def __init__(self, args):
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
        self.inhibition = np.zeros(args['inhLag'])
        self.tau = args['tau']
           
    def step(self, external):
        injection=0
        if self.name in external:
            injection = external[self.name]
        s = self.neurons.spiking.sum()
        self.pho += -self.pho/self.tau + s - self.inhibition[0] + injection
        
        #push the array
        self.inhibition[:-1] = self.inhibition[1:]
        self.inhibition[-1] = s
                
        
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
                
    
class Network():
    def __init__(self, NeuronsGroups, Connections, Modulation):

        self.NeuGroups = {}
        self.SynGroups = {}
        self.Modulators = {}
        
        #initiate neuron groups
        for groupArgs in NeuronsGroups:
            self.NeuGroups[groupArgs['label']] = groupArgs['initF'](groupArgs)
            
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
            self.SynGroups[preName+'-'+postName] = Synapses(preNeurons, postNeurons, self.Modulators, connectionArgs)     
        


    def step(self,external):
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

    def normalize(self):
        for _,connection in self.SynGroups.items():
            connection.normalize()

    def recorder(self):
        return
