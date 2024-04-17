import math
import random
import sys
import time
import heapq
import queue

from GraphClass import *

class EdgeRecord:
    def __init__(self, u, v, ux, uy, vx, vy):
        self.u = u
        self.v = v
        self.ux = ux
        self.uy = uy
        self.vx = vx
        self.vy = vy


def Generate_Geometric_Graph(n, r, fileName):
    print("--- starting generation ---")
    V = []
    edgeRecord = set()
    eR = set()
    g = Graph()
    for i in range(n):
        x = random.uniform(0,1)
        y = random.uniform(0,1)
        vertex = Vertex(str(i), x, y)
        V.append(vertex)
        g.add_vertex(vertex)

    #print("graph (just vertices):", g.printGraph())
    print(len(V), " vertices generated")
    initU = "-1"
    initV = "-2"
    init_e_record = EdgeRecord(initU, initV, -1, -1, -1, -1)
    edgeRecord.add(init_e_record)
    for u in V:
        for v in V:
            if (u != v) and (((u.x - v.x)**2 + (u.y - v.y)**2) <= (r**2)):
                #print("creating edge...", u.name, v.name)
                g.add_edge(u, v)
                eRecord = EdgeRecord(u.name,v.name, u.x, u.y, v.x, v.y)
                er = EdgeRecord(u.name,v.name, u.x, u.y, v.x, v.y)
                eR.add(er)
                isUniqueEdge = True
                for e in edgeRecord:
                    if (eRecord.u == e.u and eRecord.v == e.v) or (eRecord.u == e.v and eRecord.v == e.u):
                        isUniqueEdge = False
                if isUniqueEdge:
                    edgeRecord.add(eRecord)

    edgeRecord.remove(init_e_record)
    verticesSet = set()
    for eRecord in edgeRecord:
        verticesSet.add(eRecord.u)
        verticesSet.add(eRecord.v)
    numVertices = len(verticesSet)
    numEdges = len(edgeRecord)
    print("writing #vertices: ",  numVertices)
    print("writing #edges: ", numEdges)

    file = open(fileName + ".edges", "w")
    file.write("% sym unweighted unsigned undirected\n")
    file.write("% " + str(numEdges) + " " + str(numVertices) + " " + str(numVertices) + " " + str(numVertices) + "\n")
    for record in edgeRecord:
        #print("edge record u:", record.u, "(x:", record.ux, "y:", record.uy,") v:" ,record.v, "(x:", record.vx, "y:", record.vy, ")")
        file.write(record.u + " " + str(record.ux) + " " + str(record.uy) + " " + record.v + " " + str(record.vx) + " " + str(record.vy) + "\n")

    #print("graph (with edges):", g.printGraph())
    print("--- end of generation ---")

def Read_Geometric_Graph(fileName):
    print("--- starting reading:", fileName," ---")
    startTimer = time.time()

    g = Graph()
    vertexDict = {}
    vNames = []

    file = open(fileName, "r")
    idx = 0
    for data in file:
        if idx >= 2:
            content = data.split()
            #get data
            uName = content[0]
            uX = content[1]
            uY = content[2]
            vName = content[3]
            vX = content[4]
            vY = content[5]

            if uName not in vNames:
                u = Vertex(uName, uX, uY)
                vNames.append(uName)
                vertexDict[uName] = u
            if vName not in vNames:
                v = Vertex(vName, vX, vY)
                vNames.append(vName)
                vertexDict[vName] = v
        idx += 1
    file.close()

    file = open(fileName, "r")
    idx = 0
    for data in file:
        if idx >= 2:
            content = data.split()
            # get data
            uName = content[0]
            vName = content[3]
            g.add_edge(vertexDict[uName], vertexDict[vName])
        idx += 1
    file.close()
    print("number of vertices: ", len(g.vertices))
    print("vertex dict length:" , len(vertexDict))
    print("--- end of reading ---")
    return g

def Read_Online_Graph(fileName, startingIndex):
    print("--- starting reading:", fileName," ---")
    g = Graph()
    vertexDict = {}
    verticeNames = set()
    file = open(fileName, "r")
    idx = 0
    # read and create vertices
    for data in file:
        if idx >= startingIndex:
            # read edges
            content = data.split()
            # get data
            uName = content[0]
            vName = content[1]
            if uName not in verticeNames:
                u = Vertex(uName,0,0)
                verticeNames.add(uName)
                vertexDict[uName] = u
            if vName not in verticeNames:
                v = Vertex(vName,0,0)
                verticeNames.add(vName)
                vertexDict[vName] = v
        idx += 1
    file.close()

    #create edges by using previously created vertices
    file = open(fileName, "r")
    idx = 0
    for data in file:
        if idx >= startingIndex:
            # read edges
            content = data.split()
            # get data
            uName = content[0]
            vName = content[1]
            g.add_edge(vertexDict[uName], vertexDict[vName])
        idx += 1
    file.close()
    print("--- end of reading ---")
    return g

def DFS(g, startingVertex):
    # initialization
    for u in g.vertices:
        g.setColor(u, "white")
        g.setPi(u, "nil")
    time = 0
    if g.getColor()[startingVertex] == "white":
        DFS_Visit(g, startingVertex, time)


def DFS_Visit(g, vertexKey, time):
    time += 1
    g.setD(vertexKey, time)
    g.setColor(vertexKey, "gray")
    for neighborKey in g.vertices[vertexKey]:
        if g.getColor()[neighborKey] == "white":
            g.setPi(neighborKey, vertexKey)
            DFS_Visit(g, neighborKey, time)
    time += 1
    g.setF(vertexKey, time)
    g.setColor(vertexKey, "black")

def CalculateLCC(g):
    cc = []
    # initialization
    for u in g.vertices:
        g.setColor(u, "white")
    # starting for dfs
    for v in g.vertices:
        if g.getColor()[v] == "white":
            tmp = []
            cc.append(DfsVisitCC(g, tmp, v))

    ## cc = list consisting of connected components
    lcc = []
    for connectedComponents in cc:
        if len(connectedComponents) > len(lcc):
            lcc = connectedComponents
    return lcc

def DfsVisitCC(g, tmp, vertexKey):
    g.setColor(vertexKey, "black")
    tmp.append(vertexKey)
    for neighborKey in g.vertices[vertexKey]:
        if g.getColor()[neighborKey] == "white":
            tmp = DfsVisitCC(g, tmp, neighborKey)
    return  tmp

def TestingLCC():
    testG = Graph()
    a = Vertex("A", 0, 0)
    b = Vertex("B", 0, 1)
    c = Vertex("C", 0, 2)
    d = Vertex("D", 0, 3)
    e = Vertex("E", 0, 4)
    f = Vertex("F", 0, 5)
    g = Vertex("G", 1, 0)
    h = Vertex("H", 2, 0)
    j = Vertex("J", 3, 0)
    testG.add_vertices([a, b, c, d, e, f, g, h, j])
    testG.add_edge(a, b)
    testG.add_edge(b, c)
    testG.add_edge(c, e)
    testG.add_edge(d, f)
    testG.add_edge(f, g)
    testG.add_edge(g, h)
    testG.add_edge(h, j)
    print("--- LCC TESTING ---")
    print("graph: ",testG.printGraph())
    LCC = CalculateLCC(testG)
    print("LCC: ", LCC)
    print("--- END of LCC TESTING ---")

def Print_Simulation_Results(g):
    print("-- SIMULATION RESULTS --")
    graphLcc = CalculateLCC(g)
    vLcc = len(graphLcc)
    print("numberOfVertices:", len(g.vertices))
    print("vLcc: ", vLcc)
    deltaLcc = 0
    kLcc = 0
    sum_kV = 0
    for v in graphLcc:
        kV = len(g.vertices[v])
        sum_kV += kV
        if kV >= deltaLcc:
            deltaLcc = kV
    kLcc = (1 / vLcc) * sum_kV
    print("deltaLcc(maximum degree of nodes in LCC):", deltaLcc)
    print("kLcc(average degree of nodes in LCC): ", kLcc)
    print("LCC Graph: ",graphLcc)

    print("-- END OF SIMULATION RESULTS --")

def DFS_Based_Longest_Simple_Path(g):
    print("--- DFS Based Longest Simple Path ---")
    graphLcc = CalculateLCC(g)
    vLcc = len(graphLcc)
    Lmax = 0
    print("DFS will be run ", math.sqrt(vLcc), "times")
    for i in range(int(math.sqrt(vLcc)) + 1):
        print("DFS #", i)
        # selecting random vertex
        randomVertex = random.choice(graphLcc)
        # apply dfs
        DFS(g, randomVertex)
        # finding the key and value for the greatest depth vertex
        max_d_key_for_randomVertex = Get_Largest_D_Key(g, randomVertex)
        max_d_for_randomVertex = g.getD(max_d_key_for_randomVertex)

        # applying dfs by starting from the greatest depth vertex found before
        DFS(g, max_d_key_for_randomVertex)
        # finding the key and value for this dfs
        max_d_key_for_otherVertex = Get_Largest_D_Key(g, max_d_key_for_randomVertex)
        max_d_for_otherVertex = g.getD(max_d_key_for_otherVertex)

        # Lmax
        Lmax = max(Lmax, max_d_for_randomVertex, max_d_for_otherVertex)
    print("For DFS Based, Lmax: ", Lmax)
    print("--- End of DFS Based Longest Simple Path ---")

def Get_Largest_D_Key(g, startingVertexKey):
    max_d_key = startingVertexKey
    max_d = g.getD(max_d_key)
    for v in g.vertices:
        if g.getD(v) >= max_d and g.getD(v) != math.inf:
            max_d_key = v
            max_d = g.getD(max_d_key)
    return max_d_key

def Get_A_Source_Node(g):
    graphSize = len(g.vertices)
    number = random.randint(0, graphSize)
    index = 0
    for v in g.vertices:
        index += 1
        if index == number:
            return v

def Initialize_Single_Source_Max(g, sKey):
    for v in g.vertices:
        g.setD(v, -math.inf)
        g.setPi(v, "nil")
    g.setD(sKey,0)

def Relax_Max(g, uKey, vKey):
    if g.getD(vKey) < g.getD(uKey) + 1:
        newD = g.getD(uKey) + 1
        g.setD(vKey, newD)
        g.setPi(vKey, uKey)
        return True
    else:
        return False

def Dijskstra_Max(g, sKey):
    Initialize_Single_Source_Max(g, sKey)
    S = set()
    Q = []
    # distance, key
    for u in g.vertices:
        heappush_max(Q, (g.getD(str(u)), int(u)))
    while Q:
        ud, u = heapq._heappop_max(Q)
        S.add(str(u))
        #print("S: ", S)
        #print("calculating for adjacent nodes of vertex: ", u)
        for v in g.vertices[str(u)]:
            #print("previous d: ",g.getD(v))
            if v in S:
                #print(v," is in S!")
                continue
            #print("Q:",Q)
            index = Q.index((g.getD(str(v)), int(v)))
            result = Relax_Max(g, str(u), str(v))
            if result == True:
                Q[index] = (g.getD(str(v)), int(v)) # increase key
                heapq._heapify_max(Q)
    # below for debug purposes
    #print("results:")
    #print("source node: ", sKey)
    #for v in g.vertices:
        #print("vertex:", v, " v.d:", g.getD(v), " v.pi: ", g.getPi(v))

def heappush_max(max_heap, item):
    max_heap.append(item)
    heapq._siftdown_max(max_heap, 0, len(max_heap)-1)

def Dijkstra_Based_Longest_Simple_Path(g):
    print("--- Dijkstra Based Longest Simple Path ---")
    # we follow a similar approach with dfs longest simple path
    graphLcc = CalculateLCC(g)
    vLcc = len(graphLcc)
    Lmax = 0
    print("Dijkstra will be run ", math.sqrt(vLcc) + 1 , " times")
    for i in range(int(math.sqrt(vLcc)) + 1):
        print("Dijkstra #", i)
        # select random vertex
        randomVert = random.choice(graphLcc)
        # apply dijsktra
        Dijskstra_Max(g, randomVert)
        # find the key and value for largest d vertex
        max_d_key_for_randomVert = Get_Largest_D_Key(g, randomVert)
        max_d_for_randomVert = g.getD(max_d_key_for_randomVert)

        # this time apply dijsktra with the vertex found that was the largest d in previous run
        Dijskstra_Max(g, max_d_key_for_randomVert)
        # find the key and value for largest d vertex for the second dijkstra run
        max_d_key_for_secondRun = Get_Largest_D_Key(g, max_d_key_for_randomVert)
        max_d_for_secondRun = g.getD(max_d_key_for_secondRun)

        # calculate Lmax
        Lmax = max(Lmax, max_d_for_randomVert, max_d_for_secondRun)
    print("For Dijkstra Based, Lmax:", Lmax)
    print("--- End of Dijkstra Based Longest Simple Path ---")

def A_Star(g, sourceKey, destinationKey):
    print("- A* -")
    Initialize_Single_Source_Max(g, sourceKey)
    for v in g.vertices:
        dX = float(g.getXposition(destinationKey))
        vX = float(g.getXposition(v))
        dY = float(g.getYposition(destinationKey))
        vY = float(g.getYposition(v))
        hVal = math.sqrt((dX - vX)**2 + (dY - vY)**2)
        g.setH(v, hVal)
    S = set()
    Q = []
    for u in g.vertices:
        dh = g.getD(str(u)) + g.getH(str(u))
        heappush_max(Q, (dh, int(u)))
    #print("Q: ", Q)
    while Q:
        _, u = heapq._heappop_max(Q)
        S.add(str(u))
        #print("S: ", S)
        #print("calculating for ", u)
        for v in g.vertices[str(u)]:
            if v in S:
                continue
            dh = g.getD(str(v)) + g.getH(str(v))
            #print("dh: ", dh, "type: ", type(dh))
            #print("Q:", Q)
            index = Q.index((dh, int(v)))
            result = Relax_Max(g, str(u), str(v))
            if result:
                #print("vd is increased")
                if v in S:
                    #print("v is in S")
                    S.remove(v)
                    dh = g.getD(str(v)) + g.getH(str(v))
                    heappush_max(Q, (dh, int(v)))
                elif v not in S:
                    #print("v is not in S")
                    dh = g.getD(str(v)) + g.getH(str(v))
                    Q[index] = (dh, int(v))
                    heapq._heapify_max(Q)
    print("- End of A* -")

def A_Star_Based_Longest_Simple_Path(g):
    print("--- A* Based Longest Simple Path ---")
    # we follow a similar approach with dfs longest simple path
    graphLcc = CalculateLCC(g)
    vLcc = len(graphLcc)
    Lmax = 0
    for i in range(int(math.sqrt(vLcc)) + 1):
        # select random source and destination nodes
        s1 = random.choice(graphLcc)
        d1 = random.choice(graphLcc)
        while s1 == d1:
            d1 = random.choice(graphLcc)
        # run a*
        A_Star(g, s1, d1)
        # find the key and value for largest d vertex
        max_d_key_opt1 = Get_Largest_D_Key(g, s1)
        max_d_opt1 = g.getD(max_d_key_opt1)

        # run a* again, start with the vertex having largest d from previous run
        # select a random destination
        d2 = random.choice(graphLcc)
        while d2 == max_d_key_opt1:
            d2 = random.choice(graphLcc)
        # run a*
        A_Star(g, max_d_key_opt1, d2)
        # find the key and value for largest d vertex
        max_d_key_opt2 = Get_Largest_D_Key(g, max_d_key_opt1)
        max_d_opt2 = g.getD(max_d_key_opt2)

        #calculate Lmax
        Lmax = max(Lmax, max_d_opt1, max_d_opt2)
    print("For A* Based, Lmax:", Lmax)
    print("--- End of A* Based Longest Simple Path ---")

def BFS_Modified_For_V2(g, sKey, usedKeys):
    for u in g.vertices:
        if u in usedKeys:
            continue
        g.setColor(u, "white")
        g.setD(u, math.inf)
        g.setPi(u, "nil")
    g.setColor(sKey, "gray")
    g.setD(sKey,0)
    g.setPi(sKey, "nil")
    Q = queue.Queue() # first in first out queue, keys will be stored inside of it
    Q.put(sKey)
    while Q.qsize() > 0:
        uKey = Q.get()
        for vKey in g.vertices[str(uKey)]:
            if g.getColor()[vKey] == "white":
                g.setColor(vKey, "gray")
                newD = g.getD(uKey) + 1
                g.setD(vKey, newD)
                g.setPi(vKey, uKey)
                Q.put(vKey)
        g.setColor(uKey, "black")

def BFS(g, sKey):
    for u in g.vertices:
        g.setColor(u, "white")
        g.setD(u, math.inf)
        g.setPi(u, "nil")
    g.setColor(sKey, "gray")
    g.setD(sKey,0)
    g.setPi(sKey, "nil")
    Q = queue.Queue() # first in first out queue, keys will be stored inside of it
    Q.put(sKey)
    while Q.qsize() > 0:
        uKey = Q.get()
        for vKey in g.vertices[str(uKey)]:
            if g.getColor()[vKey] == "white":
                g.setColor(vKey, "gray")
                newD = g.getD(uKey) + 1
                g.setD(vKey, newD)
                g.setPi(vKey, uKey)
                Q.put(vKey)
        g.setColor(uKey, "black")

def BFS_Modified_For_V3(g, sKey, calculatedKeys):
    numUsed = 0
    numNotUsed = 0
    for u in g.vertices:
        if u in calculatedKeys:
            numUsed += 1
            g.setD(u, math.inf)
            continue
        numNotUsed += 1
        #print("vertex ", u," is NOT USED before.")
        g.setColor(u, "white")
        g.setD(u, math.inf)
        g.setPi(u, "nil")
    g.setColor(sKey, "gray")
    g.setD(sKey, 0)
    g.setPi(sKey, "nil")
    Q = queue.Queue()  # first in first out queue, keys will be stored inside of it
    Q.put(sKey)
    while Q.qsize() > 0:
        uKey = Q.get()
        for vKey in g.vertices[str(uKey)]:
            if g.getColor()[vKey] == "white":
                g.setColor(vKey, "gray")
                newD = g.getD(uKey) + 1
                g.setD(vKey, newD)
                g.setPi(vKey, uKey)
                Q.put(vKey)
        g.setColor(uKey, "black")

def Get_Used_Keys_For_BFS_V3(g, usedKeys, largestDkey):
    parentKey = g.getPi(largestDkey)
    if (parentKey != "nil") and (parentKey not in usedKeys):
        usedKeys.append(parentKey)
    while parentKey != "nil":
        parentKey = g.getPi(parentKey)
        if (parentKey != "nil") and (parentKey not in usedKeys):
            usedKeys.append(parentKey)
    return usedKeys


def BFS_Heuristic_V3(g):
    print("--- BFS Heuristic V3 ---")
    path = []
    graphLcc = CalculateLCC(g)
    sKey = random.choice(graphLcc)
    vLcc = len(graphLcc)
    Lmax = 0
    for v in range(int(math.sqrt(vLcc))):
        BFS_Modified_For_V3(g, sKey, path)
        highest_d_key = Get_Largest_D_Key(g, sKey)
        if highest_d_key not in path:
            path.append(highest_d_key)
        path = Get_Used_Keys_For_BFS_V3(g, path, highest_d_key)
        sKey = highest_d_key
    Lmax = len(path)
    print("For custom BFS Heuristic, Lmax:", Lmax)
    print("--- End of BFS Heuristic V3 ---")

def BFS_Heuristic_V2(g):
    print("--- BFS Heuristic V2 ---")
    # we follow a similar approach with dfs longest simple path
    graphLcc = CalculateLCC(g)
    vLcc = len(graphLcc)
    Lmax = 0
    usedKeys = []
    print("BFS Heuristic V2 will be called #",2*len(graphLcc)," times")
    numTimes = 0
    for v in graphLcc:
        numTimes += 1
        print("BFS Heuristic V2 #", numTimes)
        BFS_Modified_For_V2(g, v, usedKeys)
        usedKeys.append(v)
        max_d_key = Get_Largest_D_Key(g,v)
        numTimes += 1
        print("BFS Heuristic V2 #", numTimes)
        BFS_Modified_For_V2(g, max_d_key, usedKeys)
        max_d_key_2 = Get_Largest_D_Key(g,max_d_key)
        Lmax = max(Lmax, g.getD(max_d_key), g.getD(max_d_key_2))
    print("For BFS Heuristic V2, Lmax:", Lmax)

def BFS_Heuristic_V1(g):
    print("--- BFS Heuristic V1 ---")
    # we follow a similar approach with dfs longest simple path
    graphLcc = CalculateLCC(g)
    Lmax = 0
    numTimes = 0
    print("BFS Heuristic V1 will be called #", 2 * len(graphLcc), " times")
    for v in graphLcc:
        numTimes += 1
        print("BFS Heuristic V1 #", numTimes)
        BFS(g, v)
        max_d_key = Get_Largest_D_Key(g, v)
        numTimes += 1
        print("BFS Heuristic V1 #", numTimes)
        BFS(g, max_d_key)
        max_d_key_2 = Get_Largest_D_Key(g, max_d_key)
        Lmax = max(Lmax, g.getD(max_d_key), g.getD(max_d_key_2))
    print("For BFS Heuristic V1, Lmax:", Lmax)

# testing field
#fileName = "graph_4_n_300_r_0_08.edges"
#Generate_Geometric_Graph(40, 0.5, fileName)
#graph = Read_Geometric_Graph(fileName)
#Print_Simulation_Results(graph)
#DFS_Based_Longest_Simple_Path(graph)
#Dijkstra_Based_Longest_Simple_Path(graph)
#A_Star_Based_Longest_Simple_Path(graph)
#BFS_Heuristic_V1(graph)
#BFS_Heuristic_V2(graph)
#BFS_Heuristic_V3(graph)


#onlineGraph = Read_Online_Graph("inf-power.mtx",3)
#Print_Simulation_Results(onlineGraph)
#sys.setrecursionlimit(2000)
#DFS_Based_Longest_Simple_Path(onlineGraph)
#Dijkstra_Based_Longest_Simple_Path(onlineGraph)
#BFS_Heuristic_V1(onlineGraph)
#BFS_Heuristic_V2(onlineGraph)
#BFS_Heuristic_V3(onlineGraph)

onlineGraph = Read_Online_Graph("DSJC500-5.mtx", 2)
Print_Simulation_Results(onlineGraph)
#DFS_Based_Longest_Simple_Path(onlineGraph)
#Dijkstra_Based_Longest_Simple_Path(onlineGraph)
#BFS_Heuristic_V1(onlineGraph)
#BFS_Heuristic_V2(onlineGraph)
BFS_Heuristic_V3(onlineGraph)

#onlineGraph = Read_Online_Graph("inf-euroroad.edges",2)
#Print_Simulation_Results(onlineGraph)
#DFS_Based_Longest_Simple_Path(onlineGraph)
#Dijkstra_Based_Longest_Simple_Path(onlineGraph)
#BFS_Heuristic_V1(onlineGraph)
#BFS_Heuristic_V2(onlineGraph)
#BFS_Heuristic_V3(onlineGraph)







