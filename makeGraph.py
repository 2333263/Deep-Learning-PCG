import os
import json
import matplotlib.pyplot as plt
import numpy as np
countsForFitness={}

leveLoc="GA_Stats/"

countSteps={}

avgFitness=[]
avgTimeSinceWrite=[]
avgGensSinceWrite=[]
levels=os.listdir(leveLoc)

for i in levels:
    with open(leveLoc+i,"r") as infile:
        data=json.load(infile)
        numSteps=data["NumSteps"]
        gen=0
        for step in range(numSteps):
            if(len(avgFitness)<=step):
                avgFitness.append(0)
            avgFitness[gen]+=(int(data["Generation"+str(step)+"Fitness"]))
            if("Generation"+str(step)+"Fitness" in countsForFitness):
                countsForFitness[str(step)]+=1
            else:
                countsForFitness[str(step)]=1
            gen+=1
        gen=0
        if(str(numSteps) in countSteps):
            countSteps[str(numSteps)]+=1
        else:
            countSteps[str(numSteps)]=1
        for step in range(numSteps-1):
            if(len(avgTimeSinceWrite)<=step):
                avgTimeSinceWrite.append(0)
                avgGensSinceWrite.append(0)
            avgTimeSinceWrite[gen]+=(float(data["Generation"+str(step+1)+"Time_Since_Last_Write"]))
            avgGensSinceWrite[gen]+=(int(data["Generation"+str(step+1)+"Gens_Since_Last_Write"]))
            gen+=1
            
avgFitness=np.array(avgFitness)/len(levels)
avgTimeSinceWrite=np.array(avgTimeSinceWrite)/len(levels)
avgGensSinceWrite=np.array(avgGensSinceWrite)/len(levels)

print()
bins=list(countSteps.keys())
bins.sort()
vals=list(countSteps.values())
plt.bar(bins,vals)


#plt.plot(avgFitness)
#plt.title("Average Fitness Over Time")
plt.ylabel("Count of Levels")
plt.xlabel("Steps in Generation Process")
plt.show()