from environment import Environment
from rl_agent import RlAgent
from constants import *
import os
import time

def PresentTheProgram():
    if os.name == 'posix':
        os.system('clear')
    elif os.name == 'nt':
        os.system('cls')
    print("The program will start soon")
    time.sleep(2)
    print("Before 1000 iterations have passed,\n you will notice that the agent is getting better\n as it needs fewer steps to complete a corse.")
    time.sleep(5)
    print("When it has reached 2000 iterations, you\n get to see exactly how it acts.")
    time.sleep(5)

if __name__ == '__main__':
    size = 5
    epsilon = .5
    bufferSize = 1000000
    syncRate = 1000
    batchSize = 32
    discount = .98
    graphics = False
    environment = Environment(size,size)
    agent = RlAgent(size,size,epsilon,bufferSize,syncRate,batchSize,discount)
    i = 0
    while True:
        if i ==200:
            agent.epsilon = .1
        if i == 1000:
            agent.epsilon = .05
        if i == 2000:
            agent.epsilon = .02
            graphics = True
        steps = 0
        environment = Environment(size,size)
        while environment.state is not None:
            steps+=1
            state1 = environment.state.copy()

            if graphics:
                if os.name == 'posix':
                    os.system('clear')
                elif os.name == 'nt':
                    os.system('cls')
                print(f"step: {steps}")
                environment.ShowState()
                #agent.ShowValues(state1)
                time.sleep(.1)

            a = agent.GetAction(state1)
            r = environment.PerformAction(a)
            if environment.state is not None:
                state2 = environment.state.copy()
            else:
                state2 = None
            agent.memoryBuffer.AddTransition(state1,state2,r,a)
            agent.Train()
        print(f"steps during iteration {i}: {steps}")
        if i == 0:
            PresentTheProgram()
        i+=1
