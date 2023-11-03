import json
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import copy
import AStar
from tqdm import tqdm,trange
import sys
import matplotlib.image as mpimg
from skimage.util import img_as_ubyte
if(len(sys.argv)!=2):
    print("Usage: python3 testModel.py <modelName>")
    exit()
else:
    modelName=str(sys.argv[1])
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


def getFitness(level, forceValid):
    curr=copy.deepcopy(level)
    curr=curr.reshape(30,45)
    floor=np.ones(45)
    fitness=0
    curr[len(curr)-1]=floor 
    if forceValid:
        curr[27][0]=0
        curr[27][44]=0
        curr[28][44]=1
        curr[28][0]=1
        #or curr[28][44]!=1 or curr[28][0]!=1
    if(curr[27][0]!=0 or curr[27][44]!=0 ): #ensures that the start location and end location is valid
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

def RoundLevel(level):
    for i in range(len(level)):
        if(level[i]>=0.5):
            level[i]=1
        else:
            level[i]=0
            
    return level
def GetChunks(Level,startRow,endRow,startCol,endCol):
    copyLevel=copy.deepcopy(Level)
    copyLevel=copyLevel.reshape(30,45)
    chunks=[]
     
    for i in range(startRow,endRow):
        for j in range(startCol,endCol):
            newLevel=np.ones((12,12))
            for k in range(i-6,i+6):
                for l in range(j-6,j+6):
                    if(k>=0 and k<30 and l>=0 and l<45):
                        newLevel[k-i+6][l-j+6]=copyLevel[k][l]
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


def getStack(Level,startRow,endRow,startCol,endCol):
    copyLevel=copy.deepcopy(Level)
    copyLevel=copyLevel.reshape(30,45)
    stacks=[]
     
    for row in range(startRow,endRow):
        for col in range(startCol,endCol):
            newLevel=np.zeros((12,12))
            i_min,i_max=max(0,row-6),min(len(copyLevel),row+6)
            j_min,j_max=max(0,col-6),min(len(copyLevel[row]),col+6) 
            
            row_slice=slice(i_min-row+6,i_max-row+6)
            col_slice=slice(j_min-col+6,j_max-col+6)
            
            newLevel[row_slice,col_slice]=copyLevel[i_min:i_max,j_min:j_max]
            Stack=np.zeros((30,45,2),dtype=bool)
            Stack[:,:,0]=copyLevel
            newLevel=np.pad(newLevel,((9,9),(16,17)),"constant",constant_values=0)
            Stack[:,:,1]=newLevel
            stacks.append(Stack)
    return stacks


def getModel(modelName):
    modelLoc="Models/"+modelName
    model=tf.keras.models.load_model(modelLoc)
    return model

def applyChunkChange(level,chunks,model):
    chunk=0
    copyLevel=copy.deepcopy(level)
    for row in range(len(level)):
        for col in range(len(level[0])):
            change=model(chunks[chunk].reshape(1,12,12,1)).numpy()[0]
            change=np.argmax(change)
            if(change==0):
                copyLevel[row][col]=0
            elif(change==1):
                copyLevel[row][col]=1
            chunk+=1
    return copyLevel

def applyStackedChange(level,stacks,model):
    pos=0
    copyLevel=copy.deepcopy(level)
    for row in range(len(level)):
        for col in range(len(level[0])):
            change=model(stacks[pos].reshape(1,30,45,2)).numpy()[0]
            change=np.argmax(change)
            if(change==0):
                copyLevel[row][col]=0
            elif(change==1):
                copyLevel[row][col]=1
            pos+=1
    return copyLevel


def applyComplexChange(level,chunks,model):
    copyLevel1=copy.deepcopy(level)
    copyLevel1=copyLevel1.reshape(1,30,45)
    
    copyLevel2=copy.deepcopy(level)
    copyLevel2=copyLevel2.reshape(30,45)
    chunkPos=0
    pbar=tqdm(total=len(chunks)*len(chunks[0]))
    for row in range(len(copyLevel2)):
        for col in range(len(copyLevel2[0])):
            pbar.update(1)
            change=model([np.array(copyLevel1),np.array(chunks[chunkPos].reshape(1,12,12)),np.array([row,col]).reshape(1,2)]).numpy()[0]
            change=np.argmax(change)
            if(change==0):
                copyLevel2[row][col]=0
            elif(change==1):
                copyLevel2[row][col]=1
            chunkPos+=1
            
    return copyLevel2

def applyChangeCNN(level,model):
    copyLevel=copy.deepcopy(level)
    currLevel=copyLevel.reshape(1,30,45)
    output=model(currLevel)
    output=output.numpy()
    output=output[0]
    output=RoundLevel(output)
    output=output.reshape(30,45)
    return output

def applyChangeStandardNN(level,model):
    copyLevel=copy.deepcopy(level)
    currLevel=copyLevel.reshape(1,30*45)
    output=model(currLevel)
    output=output.numpy()
    output=output[0]
    output=RoundLevel(output)
    output=output.reshape(30,45)
    return output


def getLevelImg(Level,block,flag,mario):
    
    img=np.full((16*30,16*45,3),255)
    for i in range(len(Level)):
        for j in range(len(Level[0])):
            if(Level[i][j]==1):
                img[i*16:(i+1)*16,j*16:(j+1)*16]=block
            else:
                img[i*16:(i+1)*16,j*16:(j+1)*16]=[0,96,150]
            if(i==27 and j==0 and Level[i][j]==0):
                img[i*16:(i+1)*16,j*16:(j+1)*16]=mario
            elif(i==27 and j==0 and Level[i][j]!=0):
                img[i*16:(i+1)*16,j*16:(j+1)*16]=[255,0,0]
            elif(i==27 and j==44 and Level[i][j]==0):
                img[i*16:(i+1)*16,j*16:(j+1)*16]=flag
            elif(i==27 and j==44 and Level[i][j]!=0):
                img[i*16:(i+1)*16,j*16:(j+1)*16]=[255,0,0]
                
                
    return img
    
    # img=1-Level
    # Rgb=np.full(img.shape+(3,),255)
    # blockReplace=img==0
    # for i in range(len(blockReplace)):
    #     for j in range(len(blockReplace[0])):
    #         if(blockReplace[i][j]):
    #             Rgb[i,j]=block
    # #     Rgb[loc]=block
    # #Rgb[img==0]=block
    # #print(img==0)
    # #print(Rgb[img==0])
    # Rgb[img==1]=[0,184,184]
    # Rgb[27][0]=[0,255,0]
    # Rgb[27][44]=[0,255,0]
    # return Rgb
def main():
    global modelName
    #iterations=1
    model=getModel(modelName)
    forceValid=False
    highestFitness=-100000
    bestLevel=None
    block=mpimg.imread("Legally Distinct Block.png")
    flag=mpimg.imread("Legally Distinct Flag.png")
    mario=mpimg.imread("Legally Distinct Mario.png")
    block=img_as_ubyte(block)
    flag=img_as_ubyte(flag)
    mario=img_as_ubyte(mario)
   # block=resize(block,(2,45))
        #1,2,3,4,5,10,100,1000,10000
    for iterations in [1]:
        fitness=[]
        
        startLevel=np.random.randint(0,2,size=(30,45))
        fig,axes=plt.subplots(1,3,figsize=(15,5))
        ax=axes.ravel()
        fitness.append(getFitness(startLevel,forceValid)[0])
        currLevel=startLevel.reshape(30,45)

        
        ax[0].imshow(getLevelImg(startLevel,block,flag,mario))
        ax[0].set_title("Original Level")
        ax[0].axis("off")
        
        #for i in trange(iterations):
        for i in trange(iterations):
            if(modelName=="DQN_Balanced_All_Elements" or modelName=="DQN_Unbalanced_All_Elements"):
                output=applyChunkChange(currLevel,GetChunks(currLevel,0,30,0,45),model)
            elif(modelName=="StandardNN"):
                output=applyChangeStandardNN(currLevel,model)
            elif(modelName=="CNN"):
                output=applyChangeCNN(currLevel,model)
            elif(modelName=="DQN_Balanced_All_Elements_Stacked" or modelName=="DQN_Unbalanced_All_Elements_Stacked"):
                output=applyStackedChange(currLevel,getStack(currLevel,0,30,0,45),model)
            elif(modelName=="ComplexModel"):
                output=applyComplexChange(currLevel,GetChunks(currLevel,0,30,0,45),model)
            fitness.append(getFitness(output,forceValid)[0])
            if(fitness[-1]>highestFitness):
                highestFitness=fitness[-1]
                bestLevel=output
            currLevel=output.reshape(1,30*45)
        
        
        ax[1].imshow(getLevelImg(output.reshape(30,45),block,flag,mario))
        ax[1].set_title(str(iterations)+" Iterations of Generation")
        ax[1].axis("off")
        
        diff=getLevelDifference(startLevel,output.reshape(30,45))
        
        ax[2].imshow(diff)
        ax[2].set_title("Difference")
        ax[2].axis("off")
        print("Highest Fitness: "+str(highestFitness))
       # plt.close()
        #plt.imshow(1-bestLevel,cmap="gray") 
        #plt.show()
        # plt.show()
    #     plt.savefig("Level Graphs/"+modelName+"/"+str(iterations)+"_Iterations.pdf")
    # # plt.title("Difference between original and generated level after "+str(iterations)+" iterations")
    #     plt.close()
    
    #     plt.plot(fitness)
    #     plt.title("Fitness After "+str(iterations)+" Iterations")
    #     plt.xlabel("Generation")
    #     plt.ylabel("Fitness")
    #     plt.savefig("Level Graphs/"+modelName+"/Fitness_"+str(iterations)+"_Iterations.pdf")
    #     plt.close()
if __name__=="__main__":
    main()