import os

path="./Level Graphs/StandardNN/"

allFiles=os.listdir(path)

for file in allFiles:
    if("False" in file):
        os.rename(path+file,path+"good graphs/"+file)