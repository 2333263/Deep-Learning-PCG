import json
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import copy

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
                trainingData.append(input)
                trainingTargets.append(target)
            f.close()
    return trainingData,trainingTargets
    
def splitData(data,targets):
    thresh=int(len(data)*0.8)
    trainingData=[]
    trainingTargets=[]

    
    for i in range(thresh):
        index=np.random.randint(len(data))
        trainingData.append(data[index])
        trainingTargets.append(targets[index])
        data.pop(index)
        targets.pop(index)
        
    testingData=copy.deepcopy(data)
    testingTargets=copy.deepcopy(targets)
    return np.array(trainingData),np.array(trainingTargets),np.array(testingData),np.array(testingTargets)
    
    
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
        tf.keras.layers.Conv2D(128,(3,3),activation='relu',input_shape=(30,45,1)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(256,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(512,(3,3),activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(750, activation='relu'),
        tf.keras.layers.Dense(500, activation='relu'),
        tf.keras.layers.Dense(750, activation='relu'),
        tf.keras.layers.Dense(30*45,activation='sigmoid')
    ])
    
    model.compile(loss=tf.keras.losses.MeanSquaredError(),optimizer=tf.keras.optimizers.Adam(learning_rate=10**-4))
    history=model.fit(train_data_set,epochs=1000,validation_data=validation_data_set,verbose=2,shuffle=True,callbacks=[tf.keras.callbacks.EarlyStopping(min_delta=0.00001,restore_best_weights=True)],use_multiprocessing=True)
   # print(history)
    
    tests=model.predict(testingData)
    roundedTests=[]
    #round every value to 0 or 1
    for i in tests:
        roundedTests.append(np.round(np.array(i)))
        
    #count correct changes per level
    correctChanges=[]
    for i in range(len(roundedTests)):
        correctChanges.append(np.sum(roundedTests[i]==testingTargets[i])/len(roundedTests[i]))
    np.save("NPYs/CNN/CorrectChanges.npy",correctChanges)
    np.save("NPYs/CNN/Loss.npy",history.history['loss'])
    np.save("NPYs/CNN/ValLoss.npy",history.history['val_loss'])
    plt.plot(history.history['loss'],label="Training Loss")
    plt.plot(history.history['val_loss'],label="Validation Loss")
    plt.legend()
    plt.savefig("Graphs/CNN/Loss.png")
    model.save("Models/CNN",save_format="tf")
    
def main():
    data,targets=getData()
    trainingData,trainingTargets,testingData,testingTargets=splitData(data,targets)
    trainModel(trainingData,trainingTargets,testingData,testingTargets)
    
    
if __name__ == "__main__":
    main()
    