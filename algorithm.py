import game
import copy
# import random
# import sys

# A subclass of game that has an added method for making the best/smartest move

def refactoredA(gameObj):
        losses = 0
        for nextMoveA in gameObj.possMoves(): # loop through all the possible moves
            newGameObj = copy.deepcopy(gameObj)
            codeA = newGameObj.play(nextMoveA)
            if codeA == 10:
                continue
            elif codeA != 1:
                continue
            else:
                losses += refactoredB(newGameObj)
        return losses

def refactoredB(ghostBoardA):
        losses = 0
        for nextMoveB in ghostBoardA.possMoves():
            ghostBoardB = copy.deepcopy(ghostBoardA)
            codeB = ghostBoardB.play(nextMoveB)
            if codeB == 12: 
                losses += 1
                continue
            elif codeB != 1:
                continue
            else:
                losses += refactoredA(ghostBoardB)
        return losses

class SmartGame(game.Game):
    def smartMove(self):
        moveRepo = {}
        for nextMoveA in self.possMoves():
            # should refactor this into one function maybe

            losses = 0 # choose the move that has the least number of losses
            ghostBoardA = copy.deepcopy(self)
            codeA = ghostBoardA.play(nextMoveA)
            if codeA == 10: # if we can win go for it
                self.play(nextMoveA)
                return # do i want a return code? not right now
            elif codeA != 1:
                continue
            else:
                losses += refactoredB(ghostBoardA)
            moveRepo[nextMoveA] = losses

        bestMove = min(moveRepo.items(), key=lambda x: x[1])[0]
        self.play(bestMove)

    def __init__(self):
        super().__init__()
    # takes in the game as a argument. Returns how many losses


b = SmartGame()
print(b.normalDisplay())
b.play(1)
print(b.normalDisplay())
b.smartMove()
print(b.normalDisplay())
b.play(0)
print(b.normalDisplay())
b.smartMove()
print(b.normalDisplay())
b.play(6)
print(b.normalDisplay())
b.smartMove()
print(b.normalDisplay())
b.play(5)
print(b.normalDisplay())
b.smartMove()
print(b.normalDisplay())
b.play(8)
print(b.normalDisplay())
