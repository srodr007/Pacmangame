from __future__ import print_function
# bustersAgents.py
# ----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
import os
from builtins import range
from builtins import object

import numpy as np

import util
from game import Agent
from game import Directions
from keyboardAgents import KeyboardAgent
import inference
import busters
from random import randint


class NullGraphics(object):
    "Placeholder for graphics"
    def initialize(self, state, isBlue=False):
        pass
    def update(self, state):
        pass
    def pause(self):
        pass
    def draw(self, state):
        pass
    def updateDistributions(self, dist):
        pass
    def finish(self):
        pass

class KeyboardInference(inference.InferenceModule):
    """
    Basic inference module for use with the keyboard.
    """

    def initializeUniformly(self, gameState):
        "Begin with a uniform distribution over ghost positions."
        self.beliefs = util.Counter()
        for p in self.legalPositions: self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observe(self, observation, gameState):
        noisyDistance = observation
        emissionModel = busters.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getPacmanPosition()
        allPossible = util.Counter()
        for p in self.legalPositions:
            trueDistance = util.manhattanDistance(p, pacmanPosition)
            if emissionModel[trueDistance] > 0:
                allPossible[p] = 1.0
        allPossible.normalize()
        self.beliefs = allPossible

    def elapseTime(self, gameState):
        pass

    def getBeliefDistribution(self):
        return self.beliefs


class BustersAgent(object):
    "An agent that tracks and displays its beliefs about ghost positions."

    def __init__(self, index=0, inference="ExactInference", ghostAgents=None, observeEnable=True,
                 elapseTimeEnable=True):
        inferenceType = util.lookup(inference, globals())
        self.inferenceModules = [inferenceType(a) for a in ghostAgents]
        self.observeEnable = observeEnable
        self.elapseTimeEnable = elapseTimeEnable

    # x=[6,"East",-1, 1,1,0,1]
    # print(self.weka.predict("j48.model", x, "./training_tutorial1_classification_filter1.arff"))

    def registerInitialState(self, gameState):
        "Initializes beliefs and inference modules"
        import __main__
        self.display = __main__._display
        for inference in self.inferenceModules:
            inference.initialize(gameState)
        self.ghostBeliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
        self.firstMove = True

    def observationFunction(self, gameState):
        "Removes the ghost states from the gameState"
        agents = gameState.data.agentStates
        gameState.data.agentStates = [agents[0]] + [None for i in range(1, len(agents))]
        return gameState

    def getAction(self, gameState):
        "Updates beliefs, then chooses an action based on updated beliefs."
        # for index, inf in enumerate(self.inferenceModules):
        #    if not self.firstMove and self.elapseTimeEnable:
        #        inf.elapseTime(gameState)
        #    self.firstMove = False
        #    if self.observeEnable:
        #        inf.observeState(gameState)
        #    self.ghostBeliefs[index] = inf.getBeliefDistribution()
        # self.display.updateDistributions(self.ghostBeliefs)
        return self.chooseAction(gameState)

    def chooseAction(self, gameState):
        "By default, a BustersAgent just stops.  This should be overridden."
        return Directions.STOP


class BustersKeyboardAgent(BustersAgent, KeyboardAgent):
    "An agent controlled by the keyboard that displays beliefs about ghost positions."

    def __init__(self, index=0, inference="KeyboardInference", ghostAgents=None):
        KeyboardAgent.__init__(self, index)
        BustersAgent.__init__(self, index, inference, ghostAgents)

    def getAction(self, gameState):
        return BustersAgent.getAction(self, gameState)

    def chooseAction(self, gameState):
        return KeyboardAgent.getAction(self, gameState)

    def printLineData(self, gameState):

        # NEW INSTANCES
        state = str(gameState.getPacmanPosition()[0]) + ',' + str(gameState.getPacmanPosition()[1]) + ','  # pacman position
        state += str(gameState.data.agentStates[0].getDirection()) + ','  # pacman direction
        for i in range(0, len(gameState.data.ghostDistances)):
            if gameState.data.ghostDistances[i] == None:
                state += str(-1) + ','
            else:
                state += str(gameState.data.ghostDistances[i]) + ','  # distance to the ghosts

        if "North" in gameState.getLegalPacmanActions():
            state += str(1) + ","  # north in legal actions
        else:
            state += str(0) + ","
        if "South" in gameState.getLegalPacmanActions():
            state += str(1) + ","  # south in legal actions
        else:
            state += str(0) + ","

        if "West" in gameState.getLegalPacmanActions():
            state += str(1) + ","  # west in legal actions
        else:
            state += str(0) + ","

        if "East" in gameState.getLegalPacmanActions():
            state += str(1) + ","  # east in legal actions
        else:
            state += str(0) + ","

        return state

    """

    def printLineData(self, gameState):
        state = str(gameState.getPacmanPosition()[0]) + ','+ str(gameState.getPacmanPosition()[1]) + ',' #pacman position
        state += str(gameState.data.agentStates[0].getDirection()) + ',' #pacman direction
        state += str(gameState.getNumAgents()-1) + ',' #number of ghosts
        state += str(sum(gameState.getLivingGhosts())) + ',' #num living ghosts
        for i in range(0, len(gameState.data.ghostDistances)):
            if gameState.data.ghostDistances[i] == None:
                state += str(-1) + ','
            else:
                state += str(gameState.data.ghostDistances[i]) + ',' #distance to the ghosts 
        state += str(gameState.getScore()) + ','
        if "North" in gameState.getLegalPacmanActions():
            state+=str(1) + "," #north in legal actions
        else:
            state+=str(0)+ ","
        if "South" in gameState.getLegalPacmanActions():
            state+=str(1)+ "," #south in legal actions
        else:
            state+=str(0)+ ","

        if "West" in gameState.getLegalPacmanActions():
            state+=str(1)+ "," #west in legal actions
        else:
            state+=str(0)+ ","

        if "East" in gameState.getLegalPacmanActions():
            state+=str(1)+ "," #east in legal actions
        else:
            state+=str(0)+ ","

        state += str(gameState.getNumFood()) + ',' #num of food
        if gameState.getDistanceNearestFood() == None:
            state += str(-1)
        else:
            state += str(gameState.getDistanceNearestFood()) #distance to the nearest food
        return state
    """


    # python busters.py -p BustersKeyboardAgent -l openHunt -g RandomGhost


from distanceCalculator import Distancer
from game import Actions
from game import Directions
import random, sys

'''Random PacMan Agent'''


class RandomPAgent(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    ''' Example of counting something'''

    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if (height == True):
                    food = food + 1
        return food

    ''' Print the layout'''

    def printGrid(self, gameState):
        table = ""
        ##print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table

    def chooseAction(self, gameState):  # en vez de random, elegir la minima distancia a un objetivo
        move = Directions.STOP
        legal = gameState.getLegalActions(0)  ##Legal position from the pacman
        move_random = random.randint(0, 3)
        if (move_random == 0) and Directions.WEST in legal:  move = Directions.WEST
        if (move_random == 1) and Directions.EAST in legal: move = Directions.EAST
        if (move_random == 2) and Directions.NORTH in legal:   move = Directions.NORTH
        if (move_random == 3) and Directions.SOUTH in legal: move = Directions.SOUTH
        return move


class GreedyBustersAgent(BustersAgent):
    "An agent that charges the closest ghost."

    def registerInitialState(self, gameState):
        "Pre-computes the distance between every two points."
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    def chooseAction(self, gameState):
        """
        First computes the most likely position of each ghost that has
        not yet been captured, then chooses an action that brings
        Pacman closer to the closest ghost (according to mazeDistance!).

        #####To find the mazeDistance between any two positions, use:
        #####self.distancer.getDistance(pos1, pos2)

        To find the successor position of a position after an action:
          successorPosition = Actions.getSuccessor(position, action)

        livingGhostPositionDistributions, defined below, is a list of
        util.Counter objects equal to the position belief
        distributions for each of the ghosts that are still alive.  It
        is defined based on (these are implementation details about
        which you need not be concerned):

          1) gameState.getLivingGhosts(), a list of booleans, one for each
             agent, indicating whether or not the agent is alive.  Note
             that pacman is always agent 0, so the ghosts are agents 1,
             onwards (just as before).

          2) self.ghostBeliefs, the list of belief distributions for each
             of the ghosts (including ghosts that are not alive).  The
             indices into this list should be 1 less than indices into the
             gameState.getLivingGhosts() list.
        """

        pacmanPosition = gameState.getPacmanPosition()
        legal = [a for a in gameState.getLegalPacmanActions()]
        livingGhosts = gameState.getLivingGhosts()
        livingGhostPositionDistributions = \
            [beliefs for i, beliefs in enumerate(self.ghostBeliefs)
             if livingGhosts[i + 1]]
        return Directions.EAST


class BasicAgentAA(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        self.countActions = 0
        self.list = []

    ''' Example of counting something'''

    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if (height == True):
                    food = food + 1
        return food

    ''' Print the layout'''

    def printGrid(self, gameState):
        table = ""
        # print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table

    def printInfo(self, gameState):
        print("---------------- TICK ", self.countActions, " --------------------------")
        # Map size
        width, height = gameState.data.layout.width, gameState.data.layout.height
        print("Width: ", width, " Height: ", height)
        # Pacman position
        print("Pacman position: ", gameState.getPacmanPosition())  ############
        # Legal actions for Pacman in current position
        print("Legal actions: ", gameState.getLegalPacmanActions())  ##############
        # Pacman direction
        print("Pacman direction: ", gameState.data.agentStates[0].getDirection())  #######
        # Number of ghosts
        print("Number of ghosts: ", gameState.getNumAgents() - 1)  # ----------------
        # Alive ghosts (index 0 corresponds to Pacman and is always false)
        print("Living ghosts: ", gameState.getLivingGhosts())  #########
        # Ghosts positions
        print("Ghosts positions: ", gameState.getGhostPositions())
        # Ghosts directions
        print("Ghosts directions: ",
              [gameState.getGhostDirections().get(i) for i in range(0, gameState.getNumAgents() - 1)])
        # Manhattan distance to ghosts
        print("Ghosts distances: ", gameState.data.ghostDistances)
        # Pending pac dots
        print("Pac dots: ", gameState.getNumFood())
        # Manhattan distance to the closest pac dot
        print("Distance nearest pac dots: ", gameState.getDistanceNearestFood())
        # Map walls
        print("Map:")
        print(gameState.getWalls())
        # Score
        print("Score: ", gameState.getScore())


class BasicAgentAA(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        self.countActions = 0
        self.list = []

    ''' Example of counting something'''

    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if (height == True):
                    food = food + 1
        return food

    ''' Print the layout'''

    def printGrid(self, gameState):
        table = ""
        # print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table

    def printInfo(self, gameState):
        print("---------------- TICK ", self.countActions, " --------------------------")
        # Map size
        width, height = gameState.data.layout.width, gameState.data.layout.height
        print("Width: ", width, " Height: ", height)
        # Pacman position
        print("Pacman position: ", gameState.getPacmanPosition())  ############
        # Legal actions for Pacman in current position
        print("Legal actions: ", gameState.getLegalPacmanActions())  ##############
        # Pacman direction
        print("Pacman direction: ", gameState.data.agentStates[0].getDirection())  #######
        # Number of ghosts
        print("Number of ghosts: ", gameState.getNumAgents() - 1)  # ----------------
        # Alive ghosts (index 0 corresponds to Pacman and is always false)
        print("Living ghosts: ", gameState.getLivingGhosts())  #########
        # Ghosts positions
        print("Ghosts positions: ", gameState.getGhostPositions())
        # Ghosts directions
        print("Ghosts directions: ",
              [gameState.getGhostDirections().get(i) for i in range(0, gameState.getNumAgents() - 1)])
        # Manhattan distance to ghosts
        print("Ghosts distances: ", gameState.data.ghostDistances)
        # Pending pac dots
        print("Pac dots: ", gameState.getNumFood())
        # Manhattan distance to the closest pac dot
        print("Distance nearest pac dots: ", gameState.getDistanceNearestFood())
        # Map walls
        print("Map:")
        print(gameState.getWalls())
        # Score
        print("Score: ", gameState.getScore())

    def chooseAction(self, gameState):
        self.countActions = self.countActions + 1
        # self.printInfo(gameState)
        print(self.printLineData(gameState))
        move = Directions.STOP
        legal = gameState.getLegalActions(0)  ##Legal position from the pacman

        """
        move_random = random.randint(0,3)
        if   ( move_random == 0 ) and Directions.WEST in legal:  move = Directions.WEST
        if   ( move_random == 1 ) and Directions.EAST in legal: move = Directions.EAST
        if   ( move_random == 2 ) and Directions.NORTH in legal:   move = Directions.NORTH
        if   ( move_random == 3 ) and Directions.SOUTH in legal: move = Directions.SOUTH



        """
        pos1 = gameState.getPacmanPosition()
        minghost = 4567876567
        posghost = 0

        for i in range(0, len(gameState.data.ghostDistances)):
            pos2 = gameState.getGhostPositions()[i]
            if self.distancer.getDistance(pos1, pos2) < minghost and gameState.getLivingGhosts()[i + 1] == True:
                minghost = self.distancer.getDistance(pos1, pos2)
                posghost = pos2

        x1, y1 = pos1
        x2, y2 = posghost

        if x1 < x2 or x1 == x2:
            if y1 < y2:
                if Directions.NORTH in legal and str(gameState.data.agentStates[0].getDirection()) != "South":
                    move = Directions.NORTH
                else:
                    if Directions.EAST in legal and str(gameState.data.agentStates[0].getDirection()) != "West":
                        move = Directions.EAST
                    elif Directions.WEST in legal and str(gameState.data.agentStates[0].getDirection()) != "East":
                        move = Directions.WEST
                    elif Directions.SOUTH in legal and str(gameState.data.agentStates[0].getDirection()) != "North":
                        move = Directions.SOUTH
                    else:
                        if Directions.NORTH in legal:
                            move = Directions.NORTH
                        else:
                            if Directions.EAST in legal:
                                move = Directions.EAST
                            elif Directions.WEST in legal:
                                move = Directions.WEST
                            else:
                                move = Directions.SOUTH

            if y1 == y2:
                if Directions.EAST in legal and str(gameState.data.agentStates[0].getDirection()) != "West":
                    move = Directions.EAST
                else:
                    if Directions.NORTH in legal and str(gameState.data.agentStates[0].getDirection()) != "South":
                        move = Directions.NORTH
                    elif Directions.SOUTH in legal and str(gameState.data.agentStates[0].getDirection()) != "North":
                        move = Directions.SOUTH
                    elif Directions.WEST in legal and str(gameState.data.agentStates[0].getDirection()) != "East":
                        move = Directions.WEST
                    else:
                        if Directions.EAST in legal:
                            move = Directions.EAST
                        else:
                            if Directions.NORTH in legal:
                                move = Directions.NORTH
                            elif Directions.SOUTH in legal:
                                move = Directions.SOUTH
                            else:
                                move = Directions.WEST
            if y1 > y2:
                if Directions.SOUTH in legal and str(gameState.data.agentStates[0].getDirection()) != "North":
                    move = Directions.SOUTH
                else:
                    if Directions.EAST in legal and str(gameState.data.agentStates[0].getDirection()) != "West":
                        move = Directions.EAST
                    elif Directions.WEST in legal and str(gameState.data.agentStates[0].getDirection()) != "East":
                        move = Directions.WEST
                    elif Directions.NORTH in legal and str(gameState.data.agentStates[0].getDirection()) != "South":
                        move = Directions.NORTH
                    else:
                        if Directions.SOUTH in legal:
                            move = Directions.SOUTH
                        else:
                            if Directions.EAST in legal:
                                move = Directions.EAST
                            elif Directions.WEST in legal:
                                move = Directions.WEST
                            else:
                                move = Directions.NORTH
        if x1 > x2:
            if y1 < y2:
                if Directions.NORTH in legal and str(gameState.data.agentStates[0].getDirection()) != "South":
                    move = Directions.NORTH
                else:
                    if Directions.WEST in legal and str(gameState.data.agentStates[0].getDirection()) != "East":
                        move = Directions.WEST
                    elif Directions.EAST in legal and str(gameState.data.agentStates[0].getDirection()) != "West":
                        move = Directions.EAST
                    elif Directions.SOUTH in legal and str(gameState.data.agentStates[0].getDirection()) != "North":
                        move = Directions.SOUTH
                    else:
                        if Directions.NORTH in legal:
                            move = Directions.NORTH
                        else:
                            if Directions.WEST in legal:
                                move = Directions.WEST
                            elif Directions.EAST in legal:
                                move = Directions.EAST
                            elif Directions.SOUTH in legal:
                                move = Directions.SOUTH

            if y1 == y2:
                if Directions.WEST in legal and str(gameState.data.agentStates[0].getDirection()) != "East":
                    move = Directions.WEST
                else:
                    if Directions.NORTH in legal and str(gameState.data.agentStates[0].getDirection()) != "South":
                        move = Directions.NORTH
                    elif Directions.SOUTH in legal and str(gameState.data.agentStates[0].getDirection()) != "North":
                        move = Directions.SOUTH
                    elif Directions.EAST in legal and str(gameState.data.agentStates[0].getDirection()) != "West":
                        move = Directions.EAST
                    else:
                        if Directions.WEST in legal:
                            move = Directions.WEST
                        else:
                            if Directions.NORTH in legal:
                                move = Directions.NORTH
                            elif Directions.SOUTH in legal:
                                move = Directions.SOUTH
                            else:
                                move = Directions.EAST
            if y1 > y2:
                if Directions.SOUTH in legal and str(gameState.data.agentStates[0].getDirection()) != "North":
                    move = Directions.SOUTH
                else:
                    if Directions.WEST in legal and str(gameState.data.agentStates[0].getDirection()) != "East":
                        move = Directions.WEST
                    elif Directions.EAST in legal and str(gameState.data.agentStates[0].getDirection()) != "West":
                        move = Directions.EAST
                    elif Directions.NORTH in legal and str(gameState.data.agentStates[0].getDirection()) != "South":
                        move = Directions.NORTH
                    else:
                        if Directions.SOUTH in legal:
                            move = Directions.SOUTH
                        else:
                            if Directions.WEST in legal:
                                move = Directions.WEST
                            elif Directions.EAST in legal:
                                move = Directions.EAST
                            else:
                                move = Directions.NORTH
        return move

    def printLineData(self, gameState):

        # NEW INSTANCES
        state = str(gameState.getPacmanPosition()[0]) + ',' + str(gameState.getPacmanPosition()[1]) + ','  # pacman position
        state += str(gameState.data.agentStates[0].getDirection()) + ','  # pacman direction
        for i in range(0, len(gameState.data.ghostDistances)):
            if gameState.data.ghostDistances[i] == None:
                state += str(-1) + ','
            else:
                state += str(gameState.data.ghostDistances[i]) + ','  # distance to the ghosts

        if "North" in gameState.getLegalPacmanActions():
            state += str(1) + ","  # north in legal actions
        else:
            state += str(0) + ","
        if "South" in gameState.getLegalPacmanActions():
            state += str(1) + ","  # south in legal actions
        else:
            state += str(0) + ","

        if "West" in gameState.getLegalPacmanActions():
            state += str(1) + ","  # west in legal actions
        else:
            state += str(0) + ","

        if "East" in gameState.getLegalPacmanActions():
            state += str(1) + ","  # east in legal actions
        else:
            state += str(0) + ","

        return state


class weka(BustersAgent):
    def printLineData(self, gameState):

        # NEW INSTANCES
        state = str(gameState.getPacmanPosition()[0]) + ',' + str(
            gameState.getPacmanPosition()[1]) + ','  # pacman position
        state += str(gameState.data.agentStates[0].getDirection()) + ','  # pacman direction
        for i in range(0, len(gameState.data.ghostDistances)):
            if gameState.data.ghostDistances[i] == None:
                state += str(-1) + ','
            else:
                state += str(gameState.data.ghostDistances[i]) + ','  # distance to the ghosts

        if "North" in gameState.getLegalPacmanActions():
            state += str(1) + ","  # north in legal actions
        else:
            state += str(0) + ","
        if "South" in gameState.getLegalPacmanActions():
            state += str(1) + ","  # south in legal actions
        else:
            state += str(0) + ","

        if "West" in gameState.getLegalPacmanActions():
            state += str(1) + ","  # west in legal actions
        else:
            state += str(0) + ","

        if "East" in gameState.getLegalPacmanActions():
            state += str(1) + ","  # east in legal actions
        else:
            state += str(0) + ","

        return state

    """
    def printLineData(self, gameState):
        state = str(gameState.getPacmanPosition()[0]) + ','+ str(gameState.getPacmanPosition()[1]) + ',' #pacman position
        state +=  str(gameState.data.agentStates[0].getDirection()) + ',' #pacman direction
        state += str(gameState.getNumAgents()-1) + ',' #number of ghosts
        state += str(sum(gameState.getLivingGhosts())) + ',' #num living ghosts
        for i in range(0, len(gameState.data.ghostDistances)):
            if gameState.data.ghostDistances[i] == None:
                state += str(-1) + ','
            else:
                state += str(gameState.data.ghostDistances[i]) + ',' #distance to the ghosts 
        state += str(gameState.getScore()) + ','
        if "North" in gameState.getLegalPacmanActions():
            state+=str(1) + "," #north in legal actions
        else:
            state+=str(0)+ ","
        if "South" in gameState.getLegalPacmanActions():
            state+=str(1)+ "," #south in legal actions
        else:
            state+=str(0)+ ","

        if "West" in gameState.getLegalPacmanActions():
            state+=str(1)+ "," #west in legal actions
        else:
            state+=str(0)+ ","

        if "East" in gameState.getLegalPacmanActions():
            state+=str(1)+ "," #east in legal actions
        else:
            state+=str(0)+ ","

        state += str(gameState.getNumFood()) + ',' #num of food
        if gameState.getDistanceNearestFood() == None:
            state += str(-1)
        else:
            state += str(gameState.getDistanceNearestFood()) #distance to the nearest food

        return state
    """

    def chooseAction(self, gameState):  # en vez de random, elegir la minima distancia a un objetivo
        move = Directions.STOP
        legal = gameState.getLegalActions(0)  ##Legal position from the pacman
        x = self.printLineData(gameState).split(",")
        x.pop()
        print(x)
        # a=self.weka.predict("j48.model", x, "./training_tutorial1_classification_filter1.arff")
        # a=self.weka.predict("ibk1.model", x, "./training_tutorial1_classification_filter1.arff")
        # a=self.weka.predict("j48_2.model", x, "./NEWinstances_tutorial1.arff")
        # a=self.weka.predict("randomforest_tutorial1.model", x, "./NEWinstances_tutorial1.arff")
        # a=self.weka.predict("ibk_tutorial1.model", x, "./NEWinstances_tutorial1.arff")
        # a=self.weka.predict("j48_keyboard.model", x, "./NEWinstances_keyboard.arff")
        # a=self.weka.predict("randomforest_keyboard.model", x, "./NEWinstances_keyboard.arff")
        a = self.weka.predict("ibk_keyboard.model", x, "./NEWinstances_keyboard.arff")

        if a not in gameState.getLegalPacmanActions():

            move_random = random.randint(0, 3)
            if (move_random == 0) and Directions.WEST in legal:  move = Directions.WEST
            if (move_random == 1) and Directions.EAST in legal: move = Directions.EAST
            if (move_random == 2) and Directions.NORTH in legal:   move = Directions.NORTH
            if (move_random == 3) and Directions.SOUTH in legal: move = Directions.SOUTH
            a = move

        return a

# cd /home/ml-uc3m/Downloads
# cd pacman
# python3 busters.py -p weka -l NEWmaze -g RandomGhost

class QLearningAgent(BustersAgent):

    # Initialization
    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        self.epsilon = 0.2
        self.alpha = 0.0
        self.discount = 0.8
        self.actions = {"North": 0, "East": 1, "South": 2, "West": 3}
        if os.path.exists("qtable.txt"):
            self.table_file = open("qtable.txt", "r+")
            self.q_table = self.readQtable()
        else:
            self.table_file = open("qtable.txt", "w+")
            # "*** CHECK: NUMBER OF ROWS IN QTABLE DEPENDS ON THE NUMBER OF STATES ***"
            self.initializeQtable(24)

    def initializeQtable(self, nrows):
        "Initialize qtable"
        self.q_table = np.zeros((nrows, len(self.actions)))

    def readQtable(self):
        "Read qtable from disc"
        table = self.table_file.readlines()
        q_table = []

        for i, line in enumerate(table):
            row = line.split()
            row = [float(x) for x in row]
            q_table.append(row)

        return q_table

    def writeQtable(self):
        "Write qtable to disc"
        self.table_file.seek(0)
        self.table_file.truncate()

        for line in self.q_table:
            for item in line:
                self.table_file.write(str(item) + " ")
            self.table_file.write("\n")

    def printQtable(self):
        "Print qtable"
        for line in self.q_table:
            print(line)
        print("\n")

    def __del__(self):
        "Destructor. Invokation at the end of each episode"
        self.writeQtable()
        self.table_file.close()

    def computePosition(self, state):
        """
        Compute the row of the qtable for a given state.
        """

        "*** YOUR CODE HERE ***"

        direction = self.createAttribute(state)[0]
        if direction == 'North, NearDots':
            return 0
        if direction == 'North, FarDots':
            return 1
        if direction == 'North-East, NearDots':
            return 2
        if direction == 'North-East, FarDots':
            return 3
        if direction == 'South, NearDots':
            return 4
        if direction == 'South, FarDots':
            return 5
        if direction == 'South-East, NearDots':
            return 6
        if direction == 'South-East, FarDots':
            return 7
        if direction == 'East, NearDots':
            return 8
        if direction == 'East, FarDots':
            return 9
        if direction == 'West, NearDots':
            return 10
        if direction == 'West, FarDots':
            return 11
        if direction == 'North-West, NearDots':
            return 12
        if direction == 'North-West, FarDots':
            return 13
        if direction == 'South-West, NearDots':
            return 14
        if direction == 'South-West, FarDots':
            return 15
        if direction == 'South-West':
            return 16
        if direction == 'South':
            return 17
        if direction == 'North-West':
            return 18
        if direction == 'North':
            return 19
        if direction == 'South-East':
            return 20
        if direction == 'East':
            return 21
        if direction == 'North-East':
            return 22
        if direction == 'West':
            return 23

    def getQValue(self, state, action):

        """
            Returns Q(state,action)
            Should return 0.0 if we have never seen a state
            or the Q node value otherwise
        """
        position = self.computePosition(state)
        action_column = self.actions[action]
        return self.q_table[position][action_column]

    def computeValueFromQValues(self, state):
        """
            Returns max_action Q(state,action)
            where the max is over legal actions.  Note that if
            there are no legal actions, which is the case at the
            terminal state, you should return a value of 0.0.
        """
        legalActions = state.getLegalPacmanActions()
        if 'Stop' in legalActions: legalActions.remove("Stop")
        if len(legalActions) == 0:
            return 0
        return max(self.q_table[self.computePosition(state)])

    def computeActionFromQValues(self, state):
        """
            Compute the best action to take in a state.  Note that if there
            are no legal actions, which is the case at the terminal state,
            you should return None.
        """
        legalActions = state.getLegalPacmanActions()
        if 'Stop' in legalActions: legalActions.remove("Stop")
        if len(legalActions) == 0:
            return None

        best_actions = [legalActions[0]]
        best_value = self.getQValue(state, legalActions[0])
        for action in legalActions:
            value = self.getQValue(state, action)
            if value == best_value:
                best_actions.append(action)
            if value > best_value:
                best_actions = [action]
                best_value = value

        return random.choice(best_actions)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
        """

        # Pick Action
        legalActions = state.getLegalPacmanActions()
        if 'Stop' in legalActions: legalActions.remove("Stop")
        action = None

        if len(legalActions) == 0:
            return action

        flip = util.flipCoin(self.epsilon)

        if flip:
            return random.choice(legalActions)
        return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
            The parent class calls this to observe a
            state = action => nextState and reward transition.
            You should do your Q-Value update here

        Q-Learning update:

        if terminal_state:
        Q(state,action) <- (1-self.alpha) Q(state,action) + self.alpha * (r + 0)
        else:
        Q(state,action) <- (1-self.alpha) Q(state,action) + self.alpha * (r + self.discount * max a' Q(nextState, a'))

        """

        "*** YOUR CODE HERE ***"
        Q = self.q_table
        Qsa = self.getQValue(state, action)

        if state.getScore() < nextState.getScore():
            Q[self.computePosition(state)][self.actions[action]] = (1 - self.alpha) * Qsa + self.alpha * (reward + 0)

        else:
            best_next_action = np.argmax(Q[self.computePosition(nextState)])
            Q[self.computePosition(state)][self.actions[action]] = (1 - self.alpha) * Qsa + self.alpha * (
                        reward + self.discount * Q[self.computePosition(nextState)][best_next_action])

        return Q

    def getPolicy(self, state):
        "Return the best action in the qtable for a given state"
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        "Return the highest q value for a given state"
        return self.computeValueFromQValues(state)

    def getReward(self, state, action, nextstate):
        "Return the obtained reward"

        "*** YOUR CODE HERE ***"

        r = 0
        ghost = self.createAttribute(state)[1]

        if abs(state.getPacmanPosition()[0] - state.getGhostPositions()[ghost][0]) > abs(
                nextstate.getPacmanPosition()[0] - nextstate.getGhostPositions()[ghost][0]) or abs(
                state.getPacmanPosition()[1] - state.getGhostPositions()[ghost][1]) > abs(
                nextstate.getPacmanPosition()[1] - nextstate.getGhostPositions()[ghost][1]):
            r = 1

        if abs(state.getPacmanPosition()[0] - state.getGhostPositions()[ghost][0]) < abs(
                nextstate.getPacmanPosition()[0] - nextstate.getGhostPositions()[ghost][0]) or abs(
                state.getPacmanPosition()[1] - state.getGhostPositions()[ghost][1]) < abs(
                nextstate.getPacmanPosition()[1] - nextstate.getGhostPositions()[ghost][1]):
            r = -1

        if abs(state.getPacmanPosition()[0] - state.getGhostPositions()[ghost][0]) == abs(
                nextstate.getPacmanPosition()[0] - nextstate.getGhostPositions()[ghost][0]) and abs(
                state.getPacmanPosition()[1] - state.getGhostPositions()[ghost][1]) == abs(
                nextstate.getPacmanPosition()[1] - nextstate.getGhostPositions()[ghost][1]):
            r = 10

        return r

    def createAttribute(self, state):  # Creating the values of the attributes in order to represent the state of a tick

        atr = 0
        PosPac = state.getPacmanPosition()
        distances = []

        for i in range(0, len(state.getLivingGhosts()) - 1):
            distances.append(state.data.ghostDistances[i])

        minimum = state.data.layout.width + state.data.layout.height
        for i in distances:
            if i != None:
                if minimum > i:
                    minimum = i
        i = distances.index(minimum)

        if True in state.getLivingGhosts():
            width = state.data.layout.width
            height = state.data.layout.height
            nearDist = (width + height) / 2

            if state.getDistanceNearestFood() != None:

                if PosPac[0] - state.getGhostPositions()[i][0] < 0:

                    if PosPac[1] - state.getGhostPositions()[i][1] < 0:

                        if state.getDistanceNearestFood() < nearDist:
                            atr = 'South-West, NearDots'

                        else:
                            atr = 'South-West, FarDots'

                    elif PosPac[1] - state.getGhostPositions()[i][1] > 0:

                        if state.getDistanceNearestFood() < nearDist:
                            atr = 'North-West, NearDots'

                        else:
                            atr = 'North-West, FarDots'

                    else:

                        if state.getDistanceNearestFood() < nearDist:
                            atr = 'West, NearDots'

                        else:
                            atr = 'West, FarDots'

                elif PosPac[0] - state.getGhostPositions()[i][0] > 0:

                    if PosPac[1] - state.getGhostPositions()[i][1] < 0:

                        if state.getDistanceNearestFood() < nearDist:
                            atr = 'South-East, NearDots'

                        else:
                            atr = 'South-East, FarDots'

                    elif PosPac[1] - state.getGhostPositions()[i][1] > 0:

                        if state.getDistanceNearestFood() < nearDist:
                            atr = 'North-East, NearDots'

                        else:
                            atr = 'North-East, FarDots'

                    else:

                        if state.getDistanceNearestFood() < nearDist:
                            atr = 'East, NearDots'

                        else:
                            atr = 'East, FarDots'


                elif (PosPac[0] - state.getGhostPositions()[i][0]) == 0:

                    if PosPac[1] - state.getGhostPositions()[i][1] > 0:

                        if state.getDistanceNearestFood() < nearDist:
                            atr = 'North, NearDots'

                        else:
                            atr = 'North, FarDots'

                    elif PosPac[1] - state.getGhostPositions()[i][1] < 0:

                        if state.getDistanceNearestFood() < nearDist:
                            atr = 'South, NearDots'

                        else:
                            atr = 'South, FarDots'
            else:
                if PosPac[0] - state.getGhostPositions()[i][0] < 0:

                    if PosPac[1] - state.getGhostPositions()[i][1] < 0:
                        atr = 'South-West'

                    elif PosPac[1] - state.getGhostPositions()[i][1] > 0:
                        atr = 'North-West'

                    else:
                        atr = 'West'

                elif PosPac[0] - state.getGhostPositions()[i][0] > 0:  # east

                    if PosPac[1] - state.getGhostPositions()[i][1] < 0:
                        atr = 'South-East'

                    elif PosPac[1] - state.getGhostPositions()[i][1] > 0:
                        atr = 'North-East'

                    else:
                        atr = 'East'


                elif (PosPac[0] - state.getGhostPositions()[i][0]) == 0:

                    if PosPac[1] - state.getGhostPositions()[i][1] > 0:
                        atr = 'North'

                    elif PosPac[1] - state.getGhostPositions()[i][1] < 0:
                        atr = 'South'

        return [atr, i]


"""        
python busters.py -p QLearningAgent -l labAA1 -k 1 -t 0.01
"""


