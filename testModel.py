import json
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import copy
import AStar
def printLevel(level):
    currLevel=level.reshape(30,45)
    currLevel=np.array(currLevel,dtype=int)
    for i in currLevel:
        line=""
        for j in i:
            line+=str(j)
        print(line)
        
        
def getLevelDifference(level1,level2):
    diff=level1!=level2
    diff=diff.astype(int)
    diffRgb=np.zeros(level1.shape+(3,))
    diffRgb[diff==1]=[1,0,0]
    
    return(diffRgb)


def getFitness(level):
    curr=copy.deepcopy(level.reshape(30,45))
    floor=np.ones(45)
    fitness=0
    curr[len(curr)-1]=floor 
    if(curr[27][0]!=0 or curr[27][44]!=0 or curr[28][44]!=1 or curr[28][0]!=1): #ensures that the start location and end location is valid
        return(-120,())
    else:
        path,dif=AStar.AStar(curr,27,0,27,44)

        if(dif==-1):
            fitness=-AStar.dist(path,(27,44))
        else:
            fitness=len(path)+2*dif
    return (fitness,path)

def updateLevel(level,changes):
    for i in range(len(changes)):
        if(changes[i]>=0.5):
            if(level[i]==0):
                level[i]=1
            else:
                level[i]=0
    return level


def GetChunks(Level,startRow,endRow,startCol,endCol):
    chunks=[]
     
    for i in range(startRow,endRow):
        for j in range(startCol,endCol):
            newLevel=np.ones((12,12))
            for k in range(i-6,i+6):
                for l in range(j-6,j+6):
                    if(k>=0 and k<30 and l>=0 and l<45):
                        newLevel[k-i+6][l-j+6]=Level[k][l]
            chunks.append(newLevel)
    '''
    for i in range(len(Level)):
        for j in range(len(Level[0])):
                i_min, i_max = max(0, i - 6), min(30, i + 6)
                j_min, j_max = max(0, j - 6), min(45, j + 6)

                # Create slices for copying data
                row_slice = slice(i_min - i + 6, i_max - i + 6)
                col_slice = slice(j_min - j + 6, j_max - j + 6)

                # Create an empty newLevel array
                newLevel = np.ones((12, 12), dtype=bool)

                # Update the relevant portion of newLevel using slicing
                newLevel[row_slice, col_slice] = Level[i_min:i_max, j_min:j_max]
                chunks.append(newLevel)
    '''
    return chunks


def getModel(modelName):
    modelLoc="Models/"+modelName
    model=tf.keras.models.load_model(modelLoc)
    return model

def main():
    modelName="StandardNN"
    startLevel=np.random.randint(0,2,size=(30,45))
    
    model=getModel(modelName)
    level=copy.deepcopy(startLevel)
    
    
    
if __name__=="__main__":
    main()