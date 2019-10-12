import numpy as np
from constants import *

class Environment:
    def __init__(self,height,width):
        self.height = height
        self.width = width
        self.state = np.zeros((height, width))
        self.agentPos = np.array([np.random.randint(0,height),np.random.randint(0,width)])
        self.goalPos = np.array([np.random.randint(0,height),np.random.randint(0,width)])
        while (self.goalPos == self.agentPos).all():
            self.goalPos = np.array([np.random.randint(0,height),np.random.randint(0,width)])
        self.state[self.agentPos[0],self.agentPos[1]]=AGENT_INDICATOR
        self.state[self.goalPos[0],self.goalPos[1]]=GOAL_INDICATOR

    def ShowState(self):
        matString = ''
        stringList = ['.','o','x']
        for i in range(self.height):
            for j in range(self.width):
                matString+=f"{stringList[int(self.state[i,j])]}"
                matString += ' '
            matString += '\n'
        print(matString)

    def PerformAction(self,action):
        if action == UP:
            if self.agentPos[0] == 0:
                return WALL_REWARD
            d1=abs(self.agentPos[0]-self.goalPos[0])
            self.state[self.agentPos[0],self.agentPos[1]] = 0
            self.agentPos[0] = self.agentPos[0]-1
            self.state[self.agentPos[0],self.agentPos[1]]=AGENT_INDICATOR
            if abs(self.agentPos[0]-self.goalPos[0])<d1 and d1>1:
                return GOOD_STEP_REWARD

        if action == DOWN:
            if self.agentPos[0] == (self.height-1):
                return WALL_REWARD
            d1=abs(self.agentPos[0]-self.goalPos[0])
            self.state[self.agentPos[0],self.agentPos[1]] = 0
            self.agentPos[0] = self.agentPos[0]+1
            self.state[self.agentPos[0],self.agentPos[1]]=AGENT_INDICATOR
            if abs(self.agentPos[0]-self.goalPos[0])<d1 and d1>1:
                return GOOD_STEP_REWARD
        if action == LEFT:
            if self.agentPos[1] == 0:
                return WALL_REWARD
            d1=abs(self.agentPos[1]-self.goalPos[1])
            self.state[self.agentPos[0],self.agentPos[1]] = 0
            self.agentPos[1] = self.agentPos[1]-1
            self.state[self.agentPos[0],self.agentPos[1]]=AGENT_INDICATOR
            if abs(self.agentPos[1]-self.goalPos[1])<d1 and d1>1:
                return GOOD_STEP_REWARD
        if action == RIGHT:
            if self.agentPos[1] == (self.width-1):
                return WALL_REWARD
            d1=abs(self.agentPos[1]-self.goalPos[1])
            self.state[self.agentPos[0],self.agentPos[1]] = 0
            self.agentPos[1] = self.agentPos[1]+1
            self.state[self.agentPos[0],self.agentPos[1]]=AGENT_INDICATOR
            if abs(self.agentPos[1]-self.goalPos[1])<d1 and d1>1:
                return GOOD_STEP_REWARD

        if (self.agentPos == self.goalPos).all():
            self.state = None
            return GOAL_REWARD
        else:
            return STEP_REWARD

