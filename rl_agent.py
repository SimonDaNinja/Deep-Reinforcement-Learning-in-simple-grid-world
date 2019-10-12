from tensorflow import keras
import numpy as np
from constants import *

class MemoryBuffer:
    def __init__(self, gridHeighte, gridWidth, bufferSize):
        self.state1 = np.zeros((bufferSize, gridHeighte,gridWidth))
        self.state2 = np.zeros((bufferSize, gridHeighte,gridWidth))
        self.reward = np.zeros(bufferSize)
        self.action = np.zeros(bufferSize,dtype=int)
        self.count = 0
        self.size = bufferSize

    def AddTransition(self, state1, state2, reward, action):
        if self.count < self.size:
            self.state1[self.count,:,:] = state1
            if state2 is not None:
                self.state2[self.count,:,:] = state2
            self.reward[self.count] = reward
            self.action[self.count] = action
            self.count += 1
        else:
            i = np.random.randint(0,self.size)
            self.state1[i,:,:] = state1
            self.state2[i,:,:] = state2
            self.reward[i] = reward
            self.action[i] = action

    def GetRandomSample(self,sampleSize):
        indeces = np.random.randint(0,self.count,sampleSize)
        state1 = self.state1[indeces,:,:]
        state2 = self.state2[indeces,:,:]
        reward = self.reward[indeces]
        action = self.action[indeces]
        return state1, state2, reward, action

class RlAgent:
    def __init__(self, gridHeight, gridWidth, epsilon, memoryBufferSize, syncRate, batchSize, discount):
        self.height = gridHeight
        self.width = gridWidth
        self. qnet = self.GenerateNeuralNetworks()
        self. tnet = self.GenerateNeuralNetworks()
        self.memoryBuffer = MemoryBuffer(gridHeight, gridWidth, memoryBufferSize)
        self.epsilon = epsilon
        self.syncCount = 0
        self.syncRate = syncRate
        self.batchSize = batchSize
        self.discount = discount
        self.preAllocatedIndeces = np.arange(batchSize)

    def GetAction(self,state):
        if np.random.rand()>self.epsilon:
            qvals = self.GetQvals(state)
            return np.argmax(qvals)
        else:
            return np.random.randint(0,4)

    def ShowValues(self,state):
        qvals = self.GetQvals(state)
        print(f"Up: {qvals[0,UP]}\nDown: {qvals[0,DOWN]}\nLeft: {qvals[0,LEFT]}\nRight: {qvals[0,RIGHT]}")
                
                

    def GetQvals(self,state):
        if state.shape == (self.height,self.width):
            return self.qnet.predict(state[np.newaxis,:,:,np.newaxis])
        else:
            return self.qnet.predict(state[:,:,:,np.newaxis])

    def GetQvalsT(self,state):
        if state.shape == (self.height,self.width):
            return self.tnet.predict(state[np.newaxis,:,:,np.newaxis])
        else:
            return self.tnet.predict(state[:,:,:,np.newaxis])

    def GenerateNeuralNetworks(self):
        net = keras.Sequential([
            keras.layers.Conv2D(32,(3,3),strides=(2,2), data_format = "channels_last", input_shape = (self.height,self.width,1),activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(4)])
        net.compile(optimizer = 'SGD',
                                    loss = 'mean_squared_error')
        return net
    
    def Train(self):
        state1, state2, reward, action = self.memoryBuffer.GetRandomSample(self.batchSize)
        target = self.GetQvals(state1)
        qvals2 = self.GetQvalsT(state2)
        target[self.preAllocatedIndeces,action] = reward+np.max(qvals2,axis=1)*self.discount
        goalEntries = reward== GOAL_REWARD
        target[goalEntries,action[goalEntries]] = 0
        self.qnet.fit(state1[:,:,:,np.newaxis], target, batch_size=self.batchSize, epochs=1, verbose=0)
        self.syncCount +=1
        if self.syncCount % self.syncRate == 0:
            self.tnet.set_weights(self.qnet.get_weights())
