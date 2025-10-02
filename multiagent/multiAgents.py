# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # print(newPos)
        food_dist = [manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()]
        if food_dist:
            food_score = 1 / min(food_dist)
        else:
            food_score = 0
        # print(min(food_dist))
        return successorGameState.getScore() + food_score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # print(gameState.getLegalActions(agentIndex=0))
        # print(gameState.getNumAgents())
        # print(gameState.generateSuccessor(0, gameState.getLegalActions(agentIndex=0)[0])) #generate successor for first legal action
        # print(self.depth)
        
        def value(state, depth, agentIndex):
            if depth == self.depth:
                return self.evaluationFunction(state)
            if state.isLose() or state.isWin():
                return self.evaluationFunction(state)
            if agentIndex == 0:
                return max_value(state, depth)
            else:
                return min_value(state, depth, agentIndex)
        
        #pacman is the maximizer
        def max_value(state, depth):
            v = float('-inf')
            for action in state.getLegalActions(0):
                successor = state.generateSuccessor(0, action)
                #next turn is ghost
                v = max(v, value(successor, depth, 1))
            return v
        
        #ghosts
        def min_value(state, depth, agentIndex):
            v = float('inf')

            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                nextAgent = agentIndex + 1
                if nextAgent == state.getNumAgents():
                    v = min(v, value(successor, depth + 1, 0))  
                else:
                    v = min(v, value(successor, depth, nextAgent)) 
            return v
        
        #find best action
        bestAction = None
        bestValue = float('-inf')
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            actionValue = value(successor, 0, 1)
            if actionValue > bestValue:
                bestValue = actionValue
                bestAction = action
        return bestAction
        # util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        def value(state, depth, agentIndex, alpha, beta):
            if state.isLose() or state.isWin() or (depth == self.depth):
                return self.evaluationFunction(state)
            if agentIndex == 0:
                return max_value(state, depth, alpha, beta)
            else:
                return min_value(state, depth, agentIndex, alpha, beta)
        
        #pacman is the maximizer
        def max_value(state, depth, alpha, beta):
            v = float('-inf')
            for action in state.getLegalActions(0):
                successor = state.generateSuccessor(0, action)
                #next turn is ghost
                v = max(v, value(successor, depth, 1, alpha, beta))
                if v > beta: return v
                alpha = max(alpha, v)
            return v
        
        #ghosts
        def min_value(state, depth, agentIndex, alpha, beta):
            v = float('inf')
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                nextAgent = agentIndex + 1
                if nextAgent == state.getNumAgents():
                    v = min(v, value(successor, depth + 1, 0, alpha, beta))
                    if v < alpha: return v
                    beta = min(beta, v)  
                else:
                    v = min(v, value(successor, depth, nextAgent, alpha, beta))
                    if v < alpha: return v
                    beta = min(beta, v) 
            return v
        
        #find best action
        bestAction = None
        bestValue = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            actionValue = value(successor, 0, 1, alpha, beta)
            if actionValue > bestValue:
                bestValue = actionValue
                bestAction = action
            alpha = max(alpha, actionValue)
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        def value(state, depth, agentIndex):
            if depth == self.depth:
                return self.evaluationFunction(state)
            if state.isLose() or state.isWin():
                return self.evaluationFunction(state)
            if agentIndex == 0:
                return max_value(state, depth)
            else:
                return exp_value(state, depth, agentIndex)
        
        #pacman is the maximizer
        def max_value(state, depth):
            v = float('-inf')
            for action in state.getLegalActions(0):
                successor = state.generateSuccessor(0, action)
                #next turn is ghost
                v = max(v, value(successor, depth, 1))
            return v
        
        #ghosts
        def exp_value(state, depth, agentIndex):
            v = 0

            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                nextAgent = agentIndex + 1
                if nextAgent == state.getNumAgents():
                    p = 1 / (len(state.getLegalActions(agentIndex)))
                    v += p * value(successor, depth + 1, 0)
                else:
                    p = 1 / (len(state.getLegalActions(agentIndex)))
                    v += p * value(successor, depth, nextAgent)
            return v
        
        #find best action
        bestAction = None
        bestValue = float('-inf')
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            actionValue = value(successor, 0, 1)
            if actionValue > bestValue:
                bestValue = actionValue
                bestAction = action
        return bestAction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <prioritize closer food pellets>
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    # Useful information you can extract from a GameState (pacman.py)
    # successorGameState = currentGameState.generatePacmanSuccessor(action)
    Pos = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood()

    # newGhostStates = successorGameState.getGhostStates()
    # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # print(newPos)
    food_dist = [manhattanDistance(Pos, foodPos) for foodPos in Food.asList()]
    if food_dist:
        food_score = 1 / min(food_dist)
    else:
        food_score = 0
    # print(min(food_dist))
    return currentGameState.getScore() + food_score - 4 * len(Food.asList())
    


# Abbreviation
better = betterEvaluationFunction
