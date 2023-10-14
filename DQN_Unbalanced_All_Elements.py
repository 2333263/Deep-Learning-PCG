import json
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import copy
from sklearn.model_selection import train_test_split

'''
This DQN is trained on every in one level vs the next generation of that level
The data set is unbalanced
it has a 12x12 window
and 3 possible outputs, 0 is change to 0, 1 is change to 1 and 2 no change
it also does not take the entire level as input, only a 12x12 window
'''


LevelLoc="GA_Levels2/"

def getData():
    level_files=os.listdir(LevelLoc)
    trainingData=[]
    trainingTargets=[]
    for level in level_files:
        with open(LevelLoc+level) as f:
            data=json.load(f)
            numSteps=int(data["NumSteps"])
            for j in range(numSteps-1):
                input=np.array(data["Generation"+str(j)],dtype=bool)
                target=np.array(data["Generation"+str(j+1)],dtype=bool)
                
                
                input=np.reshape(input,(30,45))
                target=np.reshape(target,(30,45))
                
                for row in range(len(input)):
                    for col in range(len(input[row])):
                        oneHot=np.zeros(3)
                        if(input[row][col]==target[row][col]):
                            oneHot[2]=1
                        elif(target[row][col]==True):
                            oneHot[1]=1
                        else:
                            oneHot[0]=1
                        newLevel=np.zeros((12,12))
                        i_min,i_max=max(0,row-6),min(len(input),row+6)
                        j_min,j_max=max(0,col-6),min(len(input[row]),col+6)
                        
                        row_slice=slice(i_min-row+6,i_max-row+6)
                        col_slice=slice(j_min-col+6,j_max-col+6)
                        
                        newLevel[row_slice,col_slice]=input[i_min:i_max,j_min:j_max]
                        trainingData.append(newLevel)
                        trainingTargets.append(oneHot)
            f.close()
                
                
    return trainingData,trainingTargets
    
def splitData(data,targets):
    trainingData,testingData,trainingTargets,testingTargets=train_test_split(data,targets,test_size=0.2)
    return(np.array(trainingData),np.array(trainingTargets),np.array(testingData),np.array(testingTargets))
    # thresh=int(len(data)*0.8)
    # trainingData=[]
    # trainingTargets=[]

    
    # for i in tqdm(range(thresh)):
    #     index=np.random.randint(len(data))
    #     trainingData.append(data[index])
    #     trainingTargets.append(targets[index])
    #     data.pop(index)
    #     targets.pop(index)
        
    # testingData=copy.deepcopy(data)
    # testingTargets=copy.deepcopy(targets)
    # return np.array(trainingData),np.array(trainingTargets),np.array(testingData),np.array(testingTargets)
    
    
def trainModel(trainingData,trainingTargets,testingData,testingTargets):
    #create a data loader for the model
    train_loader = tf.data.Dataset.from_tensor_slices((trainingData, trainingTargets))
    num_samples=len(trainingData)
    num_validation_samples=int(num_samples*0.2)
    train_loader.shuffle(num_samples)
    train_data_set=train_loader.skip(num_validation_samples)
    validation_data_set=train_loader.take(num_validation_samples)
    batch_size=32
    train_data_set=train_data_set.shuffle(num_samples).batch(batch_size)
    validation_data_set=validation_data_set.batch(batch_size)
    
    #create the model:
    
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((12,12,1),input_shape=(12,12)),
        tf.keras.layers.Conv2D(32,(12,12),activation="relu",padding="same"),
        #tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Conv2D(64,(9,9),activation="relu",padding="same"),
        #tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Conv2D(128,(3,3),activation="relu",padding="same"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256,activation="relu"),
       # tf.keras.layers.Dense(128,activation="relu"),
        tf.keras.layers.Dense(3,activation="softmax")

    ])
    
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),optimizer=tf.keras.optimizers.Adam(learning_rate=10**-4))
    history=model.fit(train_data_set,epochs=1000,validation_data=validation_data_set,verbose=2,shuffle=True,callbacks=[tf.keras.callbacks.EarlyStopping(min_delta=0.00001,restore_best_weights=True)],use_multiprocessing=True)
   # print(history)
    
    tests=model.predict(testingData)
    #round every value to 0 or 1
    correctChanges=0
    num_tests=len(tests)
    for i in range(len(tests)):
        if(np.argmax(tests[i])==np.argmax(testingTargets[i])):
            correctChanges+=1
    information={"CorrectChanges":correctChanges,"NumTests":num_tests,"Accuracy":correctChanges/num_tests}
    with open("NPYs/DQN_Unbalanced_All_Elements/Info.json","w") as f:
        json.dump(information,f)
        f.close()
    np.save("NPYs/DQN_Unbalanced_All_Elements/Loss.npy",history.history['loss'])
    np.save("NPYs/DQN_Unbalanced_All_Elements/ValLoss.npy",history.history['val_loss'])
    plt.plot(history.history['loss'],label="Training Loss")
    plt.plot(history.history['val_loss'],label="Validation Loss")
    plt.legend()
    plt.savefig("Graphs/DQN_Unbalanced_All_Elements/Loss.png")
    model.save("Models/DQN_Unbalanced_All_Elements",save_format="tf")
    
def main():
    data,targets=getData()
    trainingData,trainingTargets,testingData,testingTargets=splitData(data,targets)
    trainModel(trainingData,trainingTargets,testingData,testingTargets)
    
    
if __name__ == "__main__":
    main()
    