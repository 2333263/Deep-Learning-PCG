import numpy as np
import heapq

class Node:
    def __init__(self,posRow,posCol):
        self.posRow=posRow
        self.posCol=posCol
        self.parent=None
        self.g=0
        self.h=0
        self.f=0
        
    def __lt__(self,other):
        return self.f<other.f
def printLevel(level):
    for i in range(len(level)):
        line=""
        for j in range(len(level[0])):
            line+=str(level[i][j])
        print(line)

def dist(start, end):
    return(abs(end[1]-start[1])+abs(end[0]-start[0])) #manhattan distance
def posOnArc(x,numBlocks,divs):
    return (-1*(x**2)+(numBlocks)*x)/divs

def traceArc(level,startRow,startCol,numBlocks):
    divs=1
    finalDestRow=0.0
    finalDestCol=0.0
    change=0
    prevRow=0
    prevCol=0
    offset=0
    if(numBlocks>=3):
        change=numBlocks-2
    while(True):
        if (np.ceil(finalDestCol)+startCol>=len(level[0])):
            if(level[startRow-int(np.floor(finalDestRow))+offset][len(level[0])-1]==0):
                if(level[startRow-int(np.floor(finalDestRow))+offset+1][len(level[0])-1]==1):
                    return(int(startRow-finalDestRow),int(len(level[0])-1))
                while(level[startRow-int(np.floor(finalDestRow))+offset][len(level[0])-1]==0):
                    finalDestRow-=1
                finalDestRow+=1
                return(int(startRow-finalDestRow),int(len(level[0])-1))
            else:
                return(-1,-1)
        prevRow=finalDestRow
        finalDestRow=int(np.floor(posOnArc(float(finalDestCol),numBlocks,divs)))
        tempCol=int(np.ceil(finalDestCol))
        if(finalDestRow<0):
            divs=10
        if(startRow-finalDestRow>=len(level)):
            if(pow(1/2,change)==0):
                return(-1,-1)#mario fell out of the world
            else:
                change+=1
                finalDestCol=prevCol+pow(1/2,change)
            continue

        if(level[startRow-finalDestRow+offset][tempCol+startCol]==0):
            level[startRow-finalDestRow+offset][tempCol+startCol]=7
            prevCol=finalDestCol
            finalDestCol+=pow(1/2,change)
            level[startRow-finalDestRow+offset][tempCol+startCol]=0
        else:
            checks=[int(np.floor(prevRow)),int(np.floor(finalDestRow)),int(np.floor(posOnArc(prevCol,numBlocks,divs)))]
            for k in checks:
                if(startRow-k>=len(level) or startRow-k+1>=len(level)):
                    return(-1,-1)
            if(change<10):
                change+=1
                finalDestCol=prevCol+pow(1/2,change)
                finalDestRow=prevRow
            else:
                prevRow=posOnArc(prevCol,numBlocks,divs)
                if( level[startRow-int(np.floor(prevRow))+1][int(np.ceil(prevCol))+startCol]==0):
                    while(level[startRow-int(np.floor(prevRow))][int(np.ceil(prevCol)+startCol)]==0):
                        prevRow-=1
                        if(startRow-int(np.floor(prevRow))>=len(level)):
                            return(-1,-1)
                    prevRow+=1
                finalDestCol=int(np.ceil(prevCol))
                finalDestRow=int(np.floor(prevRow))
                break

    return(int(startRow-finalDestRow),int(finalDestCol+startCol))
def getMoves(curr,level):
    moves=[]
    if(curr.posCol+1<len(level[0]) and level[curr.posRow][curr.posCol+1]==0):
        if(curr.posRow+1<len(level) and level[curr.posRow+1][curr.posCol+1]==1):
            moves.append((curr.posRow,curr.posCol+1))
        else:
            if(curr.posRow+1<len(level) and level[curr.posRow+1][curr.posCol+1]==0):
                offset=1
                while(True):
                    if(curr.posRow+offset>=len(level)):
                        break
                    if(level[curr.posRow+offset][curr.posCol+1]==1):
                        moves.append((curr.posRow+offset-1,curr.posCol+1))
                        break
                    elif(level[curr.posRow+offset][curr.posCol+1]!=0):
                        break
                    offset+=1
    if(curr.posCol-1>=0 and level[curr.posRow][curr.posCol-1]==0):
        if(curr.posRow+1<len(level) and level[curr.posRow+1][curr.posCol-1]==1):
            moves.append((curr.posRow,curr.posCol-1))
        else:
            if(curr.posRow+1<len(level) and level[curr.posRow+1][curr.posCol-1]==0):
                offset=1
                while(True):
                    if(curr.posRow+offset>=len(level)):
                        break
                    if(level[curr.posRow+offset][curr.posCol-1]==1):
                        moves.append((curr.posRow+offset-1,curr.posCol-1))
                        break
                    elif(level[curr.posRow+offset][curr.posCol-1]!=0):
                        break
                    offset+=1
    level[curr.posRow][curr.posCol]=-1         
    jumpSizes=[2,3,4]
    for i in jumpSizes:
        jump=traceArc(level,curr.posRow,curr.posCol,i)
        if(level[jump[0]][jump[1]]==0 and level[jump[0]+1][jump[1]]==1):
            moves.append(jump)
        
    level[curr.posRow][curr.posCol]=0
    return(moves)



def AStar(level,startPosRow,startPosCol, endRow, endCol):
    start=Node(startPosRow,startPosCol)
    open=[start]
    heapq.heapify(open)
    closed=[]
    expanded=0
    cloest=(-1,-1)
    while(len(open)>0):
        if(cloest==(-1,-1)):
            cloest=(open[0].posRow,open[0].posCol)
        curr=heapq.heappop(open)
        closed.append(curr)
        if(curr.posRow==endRow and curr.posCol==endCol):
            path=[]
            while(curr.parent!=None):
                path.append((curr.posRow,curr.posCol))
                curr=curr.parent
            return(path,expanded)
        moves=getMoves(curr,level)
        for i in moves:
            if(level[i[0]][i[1]]==0):
                expanded+=1
                child=Node(i[0],i[1])
                child.parent=curr
                child.g=curr.g+1
                child.h=dist((child.posRow,child.posCol),(endRow,endCol))
                child.f=child.g+child.h
                inClosed=False

                for i in closed:
                    if(child.posRow==i.posRow and child.posCol==i.posCol):
                        inClosed=True
                        break
                inOpen=False
                for i in open:
                    if(child.posRow==i.posRow and child.posCol==i.posCol):
                        inOpen=True
                        if(child.f<i.f):
                            open.remove(i)
                            heapq.heapify(open)
                            heapq.heappush(open,child)
                        break
                if(inClosed):
                    continue
                if(inOpen):
                    continue
                if(dist((child.posRow,child.posCol),(endRow,endCol))<dist(cloest,(endRow,endCol))):
                    cloest=(child.posRow,child.posCol)
                heapq.heappush(open,child)
    return(cloest,-1)
    
'''
level=np.zeros((30,45),dtype=int)
startPosRow=27
startPosCol=0
with open("level.txt") as f:
    for i in range(len(level)):
        line=f.readline()
        for j in range(len(level[0])):
            level[i][j]=int(line[j])
            
startRow=27
startCol=0

endRow=37
endCol=44


path,expanded=AStar(level,startPosRow,startPosCol,endRow,endCol)
if(expanded==-1):
    print("No path found")
    print(path)
else:
    level[startRow][startCol]=5
    for i in path:
        level[i[0]][i[1]]=7
        print(path)
#printLevel(level)
'''