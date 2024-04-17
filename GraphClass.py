class Vertex:
    def __init__(self, vertex, x, y):
        self.name = vertex
        self.neighbors = []
        self.x = x
        self.y = y
        self.color = "none"
        self.pi = "nil"
        self.d = 0
        self.f = 0
        self.h = 0

    def add_neighbor(self, neighbor):
        if neighbor.name not in self.neighbors:
            self.neighbors.append(neighbor.name)
            neighbor.neighbors.append(self.name)

    def add_neighbors(self, neighbors):
        for neighbor in neighbors:
            self.add_neighbor(neighbor)

class Graph:
    def __init__(self):
        self.vertices = {}
        self.xPositions = {}
        self.yPositions = {}
        self.colors = {}
        self.pis = {}
        self.ds = {}
        self.fs = {}
        self.hs = {}
        self.lccVertices = []

    def add_vertex(self, vertex):
        self.vertices[vertex.name] = vertex.neighbors
        self.xPositions[vertex.name] = vertex.x
        self.yPositions[vertex.name] = vertex.y
        self.colors[vertex.name] = vertex.color
        self.pis[vertex.name] = vertex.pi
        self.ds[vertex.name] = vertex.d
        self.fs[vertex.name] = vertex.f
        self.hs[vertex.name] = vertex.h

    def add_vertices(self, vertices):
        for vertex in vertices:
            self.add_vertex(vertex)

    def add_edge(self, vertex_from, vertex_to):
        vertex_from.add_neighbor(vertex_to)
        self.vertices[vertex_from.name] = vertex_from.neighbors
        self.vertices[vertex_to.name] = vertex_to.neighbors
        self.xPositions[vertex_from.name] = vertex_from.x
        self.xPositions[vertex_to.name] = vertex_to.x
        self.yPositions[vertex_from.name] = vertex_from.y
        self.yPositions[vertex_to.name] = vertex_to.y
        self.colors[vertex_from.name] = vertex_from.color
        self.colors[vertex_to.name] = vertex_to.color
        self.pis[vertex_from.name] = vertex_from.pi
        self.pis[vertex_to.name] = vertex_to.pi
        self.ds[vertex_from.name] = vertex_from.d
        self.ds[vertex_to.name] = vertex_to.d
        self.fs[vertex_from.name] = vertex_from.f
        self.fs[vertex_to.name] = vertex_to.f
        self.hs[vertex_from.name] = vertex_from.h
        self.hs[vertex_to.name] = vertex_to.h

    def add_edges(self, edges):
        for edge in edges:
            self.add_edge(edge[0], edge[1])

    def adjacencyList(self):
        if len(self.vertices) >= 1:
            return [str(key) + ":" + str(self.vertices[key]) for key in self.vertices.keys()]
        else:
            return dict()

    def getXposition(self, key):
        return self.xPositions[key]

    def getYposition(self, key):
        return self.yPositions[key]

    def getVertices(self):
        return self.vertices

    def getColor(self):
        return self.colors

    def setColor(self, key, color):
        self.colors[key] = color

    def getPi(self, key):
        return self.pis[key]

    def setPi(self, key, vertex):
        self.pis[key] = vertex

    def getD(self, key):
        return self.ds[key]

    def setD(self, key, newD):
        self.ds[key] = newD

    def getF(self, key):
        return self.fs[key]

    def setF(self, key, newF):
        self.fs[key] = newF

    def getH(self, key):
        return self.hs[key]

    def setH(self, key, newH):
        self.hs[key] = newH

    def printGraph(self):
        return str(self.adjacencyList())

    def removeFromGraph(self, key):
        del self.vertices[key]
        del self.xPositions[key]
        del self.yPositions[key]
        del self.colors[key]
        del self.pis[key]
        del self.ds[key]
        del self.fs[key]
        del self.hs[key]

# TESTING GRAPH CLASS
a = Vertex('A', 0, 0.456)
b = Vertex('B',0 ,0.1)
c = Vertex('C',0,0)
d = Vertex('D',0,0)
e = Vertex('E',0,0)

a.add_neighbors([b,c,e])
b.add_neighbors([a,c])
c.add_neighbors([b,d,a,e])
d.add_neighbor(c)
e.add_neighbors([a,c])
d.add_neighbor(e)

def howToGenerateGraph():
    g = Graph()
    print(g.printGraph())
    g.add_vertices([a, b, c, d, e])
    g.add_edge(a, d)
    print(g.printGraph())


def howToUseGraph(g):
    for idx, v in enumerate(g.vertices):
        print(v)
        print("x:", g.getXpositions()[v])
        print("y:", g.getYpositions()[v])
        print("adjacent vertices: ", g.vertices[v])
        print("I will select second adjacent vertice: ", g.vertices[v][1])
        print("--for that adjacent vertice--")
        print("adjX: ", g.getXpositions()[g.vertices[v][1]])
        print("adjY: ", g.getYpositions()[g.vertices[v][1]])
        print("adjNeighbors: ", g.vertices[g.vertices[v][1]])

#print(g.vertices.values())

#print("printing xPositions: ", g.getXpositions())
#print("printing yPositions: ", g.getYpositions())
