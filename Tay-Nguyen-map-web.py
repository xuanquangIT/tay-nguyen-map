import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
from simpleai.search import breadth_first, SearchProblem
import sys
import math

# Graph data for the cities and distances
graph = {
    'Kon Tum': {'Dak Ha': 10, 'Ngoc Hoi': 70, 'Pleiku': 50},
    'Dak Ha': {'Kon Tum': 10, 'Ngoc Hoi': 60},
    'Ngoc Hoi': {'Kon Tum': 70, 'Dak Ha': 60, 'Pleiku': 120},
    'Pleiku': {'Kon Tum': 50, 'An Khe': 90, 'Chu Se': 40, 'Buon Ma Thuot': 180},
    'An Khe': {'Pleiku': 90, 'Ayun Pa': 80},
    'Chu Se': {'Pleiku': 40, 'Buon Ma Thuot': 160},
    'Ayun Pa': {'An Khe': 80, 'Buon Ma Thuot': 150},
    'Buon Ma Thuot': {'Pleiku': 180, 'Da Lat': 210, 'Dak Mil': 130},
    'Dak Mil': {'Buon Ma Thuot': 130, 'Gia Nghia': 80},
    'Gia Nghia': {'Dak Mil': 80, 'Da Lat': 180},
    'Da Lat': {'Buon Ma Thuot': 210, 'Gia Nghia': 180, 'Bao Loc': 110},
    'Bao Loc': {'Da Lat': 110}
}

# Coordinates for drawing the map
coordinates = {
    'Kon Tum': (1, 9),
    'Dak Ha': (2, 8),
    'Ngoc Hoi': (1, 6),
    'Pleiku': (3, 8),
    'An Khe': (5, 8),
    'Chu Se': (3, 7),
    'Ayun Pa': (5, 7),
    'Buon Ma Thuot': (4, 5),
    'Dak Mil': (3, 4),
    'Gia Nghia': (4, 3),
    'Da Lat': (6, 2),
    'Bao Loc': (5, 1)
}

# Heuristic function (Euclidean distance)
def euclidean_heuristic(node, goal):
    x1, y1 = coordinates[node]
    x2, y2 = coordinates[goal]
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# SimpleAI problem class
class SimpleAIShortestPathProblem(SearchProblem):
    def __init__(self, start, goal):
        self.goal = goal
        super().__init__(initial_state=start)

    def actions(self, state):
        return list(graph[state].keys())

    def result(self, state, action):
        return action  # Return the action taken (which is the next city)

    def is_goal(self, state):
        return state == self.goal

    def heuristic(self, state):
        return euclidean_heuristic(state, self.goal)

# AIMA problem class
sys.path.append('./aima-python')  # Assuming the AIMA library is cloned here
from search import Problem, astar_search  # Correct import from AIMA's search module

class AIMAShortestPathProblem(Problem):
    def __init__(self, initial, goal):
        super().__init__(initial, goal)

    def actions(self, state):
        return list(graph[state].keys())

    def result(self, state, action):
        return action  # Return the next city (action taken)

    def path_cost(self, cost_so_far, A, action, B):
        return cost_so_far + graph[A][B]

    def h(self, node):
        return euclidean_heuristic(node.state, self.goal)

# Find shortest path using SimpleAI with BFS
def find_shortest_path_simpleai(start, goal):
    problem = SimpleAIShortestPathProblem(start, goal)
    result = breadth_first(problem)  # Use BFS instead of A*
    return result

# Find shortest path using AIMA AI With A*
def find_shortest_path_aima(start, goal):
    problem = AIMAShortestPathProblem(start, goal)
    result = astar_search(problem)

    if result:
        path = [start] + result.solution()  # Prepend the starting city to the solution path
        return path
    else:
        return []  # No path found

# Calculate the total distance of the shortest path
def calculate_total_distance(path):
    total_distance = 0
    for i in range(len(path) - 1):
        total_distance += graph[path[i]][path[i + 1]]
    return total_distance

# Draw the map and highlight the path
def draw_map(path, start, goal):
    G = nx.Graph()
    for city, neighbors in graph.items():
        for neighbor, distance in neighbors.items():
            G.add_edge(city, neighbor, weight=distance)

    pos = coordinates
    plt.figure(figsize=(8, 6))

    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1000, font_size=8, font_weight='semibold')

    if path:
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=3)
        nx.draw_networkx_nodes(G, pos, nodelist=path[1:-1], node_color='red', node_size=1000)

    nx.draw_networkx_nodes(G, pos, nodelist=[start], node_color='green', node_size=1200)
    nx.draw_networkx_nodes(G, pos, nodelist=[goal], node_color='blue', node_size=1200)

    for city, neighbors in graph.items():
        for neighbor, distance in neighbors.items():
            mid_x = (pos[city][0] + pos[neighbor][0]) / 2
            mid_y = (pos[city][1] + pos[neighbor][1]) / 2
            plt.text(mid_x, mid_y, f"{distance} km", fontsize=8, ha='center', color='black')

    st.pyplot(plt)

# Streamlit Interface
st.title("Tìm đường đi ngắn nhất - Tây Nguyên")

cities = list(graph.keys())

start = st.selectbox("Chọn điểm xuất phát:", cities)
goal = st.selectbox("Chọn điểm đến:", cities)
algorithm = st.radio("Chọn thuật toán:", ['SimpleAI (BFS)', 'AIMA (A*)'])

if st.button("Tìm kiếm"):
    if start == goal:
        st.error("Điểm xuất phát và điểm đến phải khác nhau!")
    else:
        if algorithm == 'SimpleAI (BFS)':
            solution = find_shortest_path_simpleai(start, goal)
            path = [node[1] for node in solution.path()] if solution else []
        else:
            path = find_shortest_path_aima(start, goal)

        if path:
            total_distance = calculate_total_distance(path)
            st.success(f"Đường đi ngắn nhất: {' -> '.join(path)}\nTổng quãng đường: {total_distance} km")
            draw_map(path, start, goal)
        else:
            st.error("Không tìm thấy đường đi ngắn nhất giữa hai điểm này.")
