import numpy as np
import tensorflow as tf
import os
import json
import copy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

'''

NOTE THIS IS BALACNED
'''
def printLevel(level):
    level=level.reshape(30,45)
    for i in level:
        line=""
        for j in i:
            line+=str(j)
        print(line)
    print()
    
    
def getLevels():
    fileLoc="GA_Levels2"
    levels=os.listdir(fileLoc)
    inputs,targets=[],[]
    numSame=0
    total=0
    inp1=[]
    inp2=[]
    inp3=[]
    for i in levels:
        with open(fileLoc+"/"+i) as f:
            data=json.load(f)
            numSteps=data["NumSteps"]
            for j in range(numSteps-1):
                
                currLevel=np.array(data["Generation"+str(j)],dtype=bool)
                nextLevel=np.array(data["Generation"+str(j+1)],dtype=bool)
                
                if((currLevel==nextLevel).all()):
                    continue
                
                currLevel=currLevel.reshape((30,45))
                nextLevel=nextLevel.reshape((30,45))
                for k in range(len(currLevel)):
                    for l in range(len(currLevel[0])):
                        total+=1
                        if(currLevel[k][l]==nextLevel[k][l]):
                            if(np.random.random()<0.5):
                                break
                            numSame+=1
                        #inp=np.zeros((30,45,2),dtype=bool)
                        inp=[]
                        temp1=np.array(copy.deepcopy(currLevel),dtype=bool)
                        temp1=temp1.reshape(30,45,1)
                        inp1.append(temp1)
                        oneHot=np.zeros(3,dtype=bool)
                        i_min, i_max = max(0, k - 6), min(30, k + 6)
                        j_min, j_max = max(0, l - 6), min(45, l + 6)

                        # Create slices for copying data
                        row_slice = slice(i_min - k + 6, i_max - k + 6)
                        col_slice = slice(j_min - l + 6, j_max - l + 6)

                        # Create an empty newLevel array
                        newLevel = np.ones((12, 12), dtype=bool)

                        # Update the relevant portion of newLevel using slicing
                        newLevel[row_slice, col_slice] = currLevel[i_min:i_max, j_min:j_max]
                        #newLevel=np.pad(newLevel,((9,9),(16,17)),"constant",constant_values=0)
                        newLevel=newLevel.reshape(12,12,1)
                        inp2.append(copy.deepcopy(newLevel)) 
                        inp3.append([k,l])
                        #inp[:,:,1]=copy.deepcopy(newLevel)
                        #inputs.append(inp)
                        
                        if(currLevel[k][l]==nextLevel[k][l]):
                            oneHot[2]=True
                        elif(nextLevel[k][l]==0):
                            oneHot[0]=True
                        elif(nextLevel[k][l]==1):
                            oneHot[1]=True
                        
                            
                        targets.append(oneHot)
            f.close()
    print("Number of same:",numSame)
    print("Total:",total)
    return(inp1,inp2,inp3,targets)
                        
def GetDataSplit(data1,data2,data3,targets):
    
    trainingData1=[]
    trainingData2=[]
    trainingData3=[]
    testingData1=[]
    testingData2=[]
    testingData3=[]
    trainingTargets=[]
    testingTargets=[]
    
    thresh=int(0.8*len(data1))
    for i in range(thresh):
        index=np.random.randint(0,len(data1))
        trainingData1.append(data1[index])
        trainingData2.append(data2[index])
        trainingData3.append(data3[index])
        
        trainingTargets.append(targets[index])
        data1.pop(index)
        data2.pop(index)
        data3.pop(index)
        targets.pop(index)
        
    testingData1=copy.deepcopy(data1)
    testingData2=copy.deepcopy(data2)
    testingData3=copy.deepcopy(data3)
    testingTargets=copy.deepcopy(targets)
    return(trainingData1,trainingData2,trainingData3,trainingTargets,testingData1,testingData2,testingData3,testingTargets)
    
    
    #trainingData,testingData,trainingTargets,testingTargets=train_test_split(data,targets,test_size=0.2)
    #return(np.array(trainingData),np.array(trainingTargets),np.array(testingData),np.array(testingTargets))

def preprocess_data(inp1,inp2,inp3,targets):
    return((inp1,inp2,inp3),targets)

def MakeModel(trainingData1,trainingData2,trainingData3,trainingTargets,testingData1,testingData2,testingData3,testingTargets):

    #trainingData,trainingTargets,testingData,testingTargets=GetDataSplit(data,targets)
    tD=tf.data.Dataset.from_tensor_slices((trainingData1,trainingData2,trainingData3,trainingTargets))
    tD=tD.map(preprocess_data)
    numSamples=len(trainingData1)
    num_vali_samples=int(numSamples*0.2)
    
    trainDataSet=tD.skip(num_vali_samples)
    valiDataSet=tD.take(num_vali_samples)
    batch_size=32
    trainDataSet=trainDataSet.shuffle(buffer_size=numSamples).batch(batch_size)
    valiDataSet=valiDataSet.batch(batch_size)
    inputs1=tf.keras.layers.Input(shape=(30,45,1))
    inputs2=tf.keras.layers.Input(shape=(12,12,1))
    input3=tf.keras.layers.Input(shape=(2,))
    x=tf.keras.layers.Conv2D(32,(12,12),activation="relu",padding="same")(inputs1)
    x=tf.keras.layers.Conv2D(64,(9,9),activation="relu",padding="same")(x)
    x=tf.keras.layers.Conv2D(128,(3,3),activation="relu",padding="same")(x)
    x=tf.keras.layers.Flatten()(x)
    x=tf.keras.Model(inputs=inputs1,outputs=x)
    
    y=tf.keras.layers.Conv2D(32,(3,3),activation="relu",padding="same")(inputs2)
    y=tf.keras.layers.Conv2D(64,(3,3),activation="relu",padding="same")(y)
    y=tf.keras.layers.Conv2D(128,(3,3),activation="relu",padding="same")(y)
    y=tf.keras.layers.Flatten()(y)
    y=tf.keras.Model(inputs=inputs2,outputs=y)
    
    combined=tf.keras.layers.concatenate([x.output,y.output,input3])
    
    z=tf.keras.layers.Dense(256,activation="relu")(combined)
    z=tf.keras.layers.Dense(3,activation="softmax")(z)
    
    model=tf.keras.Model(inputs=[x.input,y.input,input3],outputs=z)

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),optimizer=tf.keras.optimizers.Adam(learning_rate=10**-4))
    history=model.fit(trainDataSet,epochs=1000,validation_data=valiDataSet,verbose=2,shuffle=True,callbacks=[tf.keras.callbacks.EarlyStopping(min_delta=0.00001,restore_best_weights=True)],use_multiprocessing=True)
    
    #testingData1=np.reshape(testingData1,(-1,30,45,1))
    testingD=tD=tf.data.Dataset.from_tensor_slices((testingData1,testingData2,testingData3,testingTargets))
    testingD=testingD.map(preprocess_data)
    
    testingD=testingD.batch(batch_size)
    #loss,acc=model.evaluate(testingD)
    outputs=model.predict(testingD)
    print(outputs)
   # outputs=model.predict(testingData)
    #outputs=model(testingData).numpy()
    countSame=0
    for i in range(len(outputs)):
        if (np.argmax(outputs[i])==np.argmax(testingTargets[i])):
            countSame+=1
    num_tests=len(outputs)
    information={"CorrectChanges":countSame,"NumTests":num_tests,"Accuracy":countSame/num_tests}
    print("Number of outputs that are different:",countSame)
    print("Number of tests:",len(outputs))
    model.save("Models/ComplexModel",save_format="tf")
    
    np.save("NPYs/ComplexModel/Loss.npy",history.history['loss'])
    np.save("NPYs/ComplexModel/ValLoss.npy",history.history['val_loss'])
    plt.plot(history.history['loss'],label="Training Loss")
    plt.plot(history.history['val_loss'],label="Validation Loss")
    plt.legend()
    plt.savefig("Graphs/ComplexModel/Loss.png")
                        
def main():
    data1,data2,data3,targets=getLevels()
    trainingData1,trainingData2,trainingData3,trainingTargets,testingData1,testingData2,testingData3,testingTargets=GetDataSplit(data1,data2,data3,targets)
    MakeModel(trainingData1,trainingData2,trainingData3,trainingTargets,testingData1,testingData2,testingData3,testingTargets)
    #MakeModel(data,targets)
    
if __name__=="__main__":
    main()