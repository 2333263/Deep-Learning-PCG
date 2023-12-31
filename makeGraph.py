import os
import json
import matplotlib.pyplot as plt
import numpy as np
countsForFitness={}

leveLoc="GA_Stats/"

countSteps={}

avgFitness={}
avgTimeSinceWrite={}
avgGensSinceWrite={}
levels=os.listdir(leveLoc)
allfitness=[]
allTimeSinceWrite=[]
allGensSinceWrite=[]
aveFitnessPercent=np.zeros(100)
aveGenerationsPercent=np.zeros(100)

for i in levels:
    with open(leveLoc+i,"r") as infile:
        data=json.load(infile)
        numSteps=data["NumSteps"]
        gen=0

        for step in range(numSteps):
            if(len(allfitness)<=step):
                allfitness.append(0)
            allfitness[gen]+=(int(data["Generation"+str(step)+"Fitness"]))
            
            
            
            
            
            if(numSteps not in avgFitness):
                avgFitness[numSteps]=[]
            if(len(avgFitness[numSteps])<=step):
                avgFitness[numSteps].append(0)
            avgFitness[numSteps][gen]+=(int(data["Generation"+str(step)+"Fitness"]))
            pos=int((step/numSteps)*100)
            aveFitnessPercent[pos]+=int(data["Generation"+str(step)+"Fitness"])
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
            
            if(len(allTimeSinceWrite)<=step):
                allTimeSinceWrite.append(0)
                allGensSinceWrite.append(0)
            allTimeSinceWrite[gen]+=(float(data["Generation"+str(step+1)+"Time_Since_Last_Write"]))
            allGensSinceWrite[gen]+=(int(data["Generation"+str(step+1)+"Gens_Since_Last_Write"]))
            pos=int((step/numSteps)*100)
            aveGenerationsPercent[pos]+=int(data["Generation"+str(step+1)+"Gens_Since_Last_Write"])
            
            if(numSteps not in avgTimeSinceWrite):
                avgTimeSinceWrite[numSteps]=[]
                avgGensSinceWrite[numSteps]=[]
            if(len(avgTimeSinceWrite[numSteps])<=step):
                avgTimeSinceWrite[numSteps].append(0)
                avgGensSinceWrite[numSteps].append(0)
            avgTimeSinceWrite[numSteps][gen]+=(float(data["Generation"+str(step+1)+"Time_Since_Last_Write"]))
            avgGensSinceWrite[numSteps][gen]+=(int(data["Generation"+str(step+1)+"Gens_Since_Last_Write"]))
            gen+=1
#print(countSteps)
for i in avgFitness.keys():
    avgFitness[i]=np.array(avgFitness[i])/countSteps[str(i)]
    
for i in avgTimeSinceWrite.keys():
    avgTimeSinceWrite[i]=np.array(avgTimeSinceWrite[i])/countSteps[str(i)]
    avgGensSinceWrite[i]=np.array(avgGensSinceWrite[i])/countSteps[str(i)]


aveFitnessPercent=aveFitnessPercent/len(levels)
aveGenerationsPercent=aveGenerationsPercent/len(levels)

# plt.plot(aveFitnessPercent)
# plt.show()

# print(aveFitnessPercent)
# plt.plot(aveFitnessPercent)
# plt.title("Average Fitness Over Time")
# plt.ylabel("Average Fitness")
# plt.xlabel("Number of Steps in Generation Process")
# plt.show()


# #sort avgFitmess keys
# keys=list(avgFitness.keys())
# keys=sorted(keys)

# for i in keys:
#     #if(countSteps[str(i)]>40):
#         plt.plot(avgFitness[i],label="Saved Steps in Genration: "+str(i))
# plt.title("Average Fitness Over Time")
# plt.ylabel("Average Fitness")
# plt.xlabel("Number of Steps in Generation Process")
# plt.legend()
# #plt.ylim(top=0)
# #plt.savefig("./Graphs/Average Fitness Over Time Clipped.pdf")
# plt.show()


# keys=list(avgTimeSinceWrite.keys())
# keys=sorted(keys)
# for i in keys:
#     if(countSteps[str(i)]>40):
#         plt.plot(avgTimeSinceWrite[i],label="Saved Steps in Genration: "+str(i))
# plt.title("Average Time Between Writes")
# plt.ylabel("Average Time Between Writes")
# plt.xlabel("Number of Steps in Generation Process")
# plt.legend()
# plt.show()

keys=list(avgGensSinceWrite.keys())
keys=sorted(keys)
for i in keys:
    #if(countSteps[str(i)]>40):
    plt.plot(avgGensSinceWrite[i],label="Saved Steps in Genration: "+str(i))
plt.title("Average Generations Between Writes")
plt.ylabel("Average Generations Between Writes")
plt.xlabel("Number of Steps in Generation Process")
plt.legend()
#plt.savefig("./Graphs/GenerationsSinceLastWrite_outliers_removed_with_outliers.pdf")
plt.show()

# allfitness=np.array(allfitness)/len(levels)
# plt.plot(allfitness)
# plt.title("Average Fitness Over Time")
# plt.ylabel("Average Fitness")
# plt.xlabel("Number of Steps in Generation Process")  
# plt.show()

# allTimeSinceWrite=np.array(allTimeSinceWrite)/len(levels)
# plt.plot(allTimeSinceWrite)
# plt.title("Average Time Between Writes")
# plt.ylabel("Average Time Between Writes")
# plt.xlabel("Number of Steps in Generation Process")
# plt.show()

# allGensSinceWrite=np.array(allGensSinceWrite)/len(levels)
# plt.plot(allGensSinceWrite)
# plt.title("Average Generations Between Writes")
# plt.ylabel("Average Generations Between Writes")
# plt.xlabel("Number of Steps in Generation Process")
# plt.show()


# bins=list(countSteps.keys())
# bins.sort(key=int)
# newVals=[]
# for i in bins:
#     newVals.append(countSteps[i])

# vals=list(countSteps.values())
# plt.bar(bins,newVals)
# plt.ylabel("Count of Levels")
# plt.xlabel("Number of Saved Steps in the Generation Process")
# 
#plt.show()
#plt.plot(avgFitness)
#plt.title("Average Fitness Over Time")
#plt.ylabel("Count of Levels")
#plt.xlabel("Number of Steps in Generation Process")
#plt.savefig("./Graphs/Bar graph of num Steps per level.png")
#plt.show()


# leveLoc="GA_Levels2/"
# levels=os.listdir(leveLoc)
# aveGenTime=0
# aveGens=0
        
# for i in levels:
#     with open(leveLoc+i,"r") as infile:
#         data=json.load(infile)
#         aveGenTime+=float(data["GenerationTime"])
#         aveGens+=int(data["NumGens"])
        
# aveGenTime=aveGenTime/len(levels)
# aveGens=aveGens/len(levels)
# print(aveGenTime)
# print(aveGens)