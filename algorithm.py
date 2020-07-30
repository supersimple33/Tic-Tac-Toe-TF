import game
import copy
# import random
# import sys

# A subclass of game that has an added method for making the best/smartest move
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
                # for nextMoveB in ghostBoardA.possMoves():
                #     ghostBoardB = copy.deepcopy(ghostBoardA)
                #     codeB = ghostBoardB.play(nextMoveB)
                #     if codeB == 12: # if we lose then add one to this move sets lose number and move to the next game
                #         losses += 1
                #         continue
                #     elif codeB != 1:
                #         continue
                #     else:
                #         f = 1 # placeholder repeat all of ghostBoardA as ghostBoardC
                #         #
                #         # 
                #         #
            moveRepo[nextMoveA] = losses

        bestMove = min(moveRepo.items(), key=lambda x: x[1])[0]
        self.play(bestMove)

    def __init__(self):
        self.board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]] # -1, 0, 1 will represent x, blanks, and o
        self.pTurn = False
    
    # takes in the game as a argument. Returns how many losses
    def refactoredA(gameObj):
        losses = 0
        newGameObj = copy.deepcopy(gameObj)
        for nextMoveA in newGameObj.possMoves(): # loop through all the possible moves
            codeA = ghostBoardA.play(nextMoveA)
            if codeA == 10:
                #we won
            elif codeA != 1:
                continue
            else:
                losses += refactoredB(newGameObj)
        return losses
        

    def refactoredB(game):


b = SmartGame()
print(b.normalDisplay())
b.play(1)
print(b.normalDisplay())
b.smartMove()
print(b.normalDisplay())
b.play(2)
print(b.normalDisplay())
b.smartMove()
print(b.normalDisplay())
b.play(4)
print(b.normalDisplay())
b.smartMove()
print(b.normalDisplay())
