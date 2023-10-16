import pygad
import numpy as np
import AStar
import os
import time
import json
import sys


st=time.time()

#threads=6
'''
Genes:
0=empty spot
1=block
2=coin
3=enemy

'''


'''
Problem 1, pygad doesnt like 2 arrays
       solution: do multiple GAS
       GA1=height of the ground
       GA2= position of floating blocks
'''

'''
size of level= 30 blocks tall, 45 blocks wide
'''
'''
another IDEA, generate chunks of the level at a time and then append them together when calculating fitness
'''

inputs=np.ones(30*45)
desired_output = 1800

def printSoluition(solution):
    for i in solution:
        line=""
        for j in i:
            line+=str(j)
        print(line)
def fitness_func(ga_instance, solution, solution_idx):
    curr=solution.reshape(30,45)
    #for i in range(len(curr[0])):
     #   curr[len(curr)-1][i]=1
    floor=np.ones(45)
    curr[len(curr)-1]=floor 
    fitness=0
    if(curr[27][0]!=0 or curr[27][44]!=0 or curr[28][44]!=1 or curr[28][0]!=1): #ensures that the start location and end location is valid
        fitness=-120
    else:
        path,dif=AStar.AStar(curr,27,0,27,44)

        if(dif==-1):
            fitness=-AStar.dist(path,(27,44))
        else:
            fitness=len(path)+2*dif
    return fitness
numGens=1

Data={ 
}
lastGen=0
prevFitness=-99999
def on_generation(ga_instance):
    global Data
    global numGens
    global prevFitness
    global lastGen
    global startTime
    if(numGens==1):
        randParent=np.random.randint(0,16)
        parents=ga_instance.last_generation_parents[randParent]
        #floor=np.ones(45)
        #parents[len(parents)-1]=floor
        #parents=parents.reshape(30*45)
        parentFitness=ga_instance.last_generation_fitness[ga_instance.last_generation_parents_indices[randParent]]
        #Data.update({"Generation0":np.ndarray.tolist(parents)})
        Data.update({"Generation0Fitness":int(parentFitness)})    
        startTime=time.time()
    if(ga_instance.best_solution()[1]!=prevFitness):
        curr=ga_instance.best_solution()[0].reshape(30,45)
        #floor=np.ones(45)
        #curr[len(curr)-1]=floor
        #curr=curr.reshape(30*45)
        #Data.update({"Generation"+str(numGens):np.ndarray.tolist(curr)})
        Data.update({"Generation"+str(numGens)+"Fitness":int(ga_instance.best_solution()[1])})
        Data.update({"Generation"+str(numGens)+"Gens_Since_Last_Write":ga_instance.generations_completed-lastGen})
        currTime=time.time()
        Data.update({"Generation"+str(numGens)+"Time_Since_Last_Write":currTime-startTime})
        startTime=currTime
        lastGen=ga_instance.generations_completed
        numGens+=1
    prevFitness=ga_instance.best_solution()[1]
    #initial.append(ga_instance.best_solution()[0])
    curr=ga_instance.best_solution()[0].reshape(30,45)
    #printSoluition(curr)
    #print("Fitness: ",ga_instance.best_solution()[1])
    #print("Generation: ",ga_instance.generations_completed)
    
    #if(ga_instance.generations_completed%100==0):
    #    ga_instance.save("CheckPoint")
    #    return "stop"
    #print(np.abs(desired_output-ga_instance.best_solution()[1]))
    if(ga_instance.best_solution()[1]>=0):
        #print(ga_instance.population)
        return "stop"

fitness_function = fitness_func

num_generations = 1000000
num_parents_mating = 16

sol_per_pop = 50
num_genes = len(inputs)

init_range_low = 0
init_range_high = 2

parent_selection_type = "rws"
keep_parents = 6

crossover_type = "two_points"

mutation_type = "scramble"
mutation_percent_genes = 30

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       gene_type=int,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_by_replacement=True,
                       random_mutation_max_val=init_range_high,
                       random_mutation_min_val=init_range_low,
                       mutation_percent_genes=mutation_percent_genes,
                       parallel_processing=None, on_generation=on_generation, keep_elitism=15,mutation_probability=0.1)


ga_instance.run()




#pop=pop.reshape((30,45))

#printSoluition(pop)
#print()
#printSoluition(initial)
        
#if(os.path.isfile("CheckPoint.pkl")):
#    print("I ran")
#    ga_instance=pygad.load("CheckPoint")
#    ga_instance.run()
#else:


solution, solution_fitness, solution_idx = ga_instance.best_solution()


solution = np.reshape(np.array(solution), (30,45))
    #write the output to a text file
floor=np.ones(45)
solution[len(solution)-1]=floor
#et=time.time()
#elapsed=et-st
if(ga_instance.best_solution()[1]>=0):
    Data.update({"NumSteps":numGens})
    #Data.update({"GenerationTime":elapsed})
    Data.update({"NumGens":ga_instance.generations_completed})
    Levels_File="GA_Stats"

    numLevels=len(os.listdir(Levels_File))
    outJson=json.dumps(Data,indent=4)

    with open(Levels_File+"/Level"+str(numLevels)+".json","w") as outfile:
        outfile.write(outJson)

    #print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
