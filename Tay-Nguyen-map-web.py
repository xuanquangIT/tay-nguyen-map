import streamlit as st
import plotly.graph_objects as go
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

# Draw the map and show distances for all paths
def draw_map(path, start, goal):
    # Create figure
    fig = go.Figure()

    # Add edges for the entire graph and display distances
    for city, neighbors in graph.items():
        for neighbor, distance in neighbors.items():
            x0, y0 = coordinates[city]
            x1, y1 = coordinates[neighbor]
            
            # Draw lines between cities
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1],
                mode="lines",
                line=dict(color="lightblue", width=2),
                hoverinfo='none'
            ))

            # Calculate the midpoint and add the distance text
            midpoint_x = (x0 + x1) / 2
            midpoint_y = (y0 + y1) / 2
            fig.add_trace(go.Scatter(
                x=[midpoint_x], y=[midpoint_y],
                text=[f'{distance} km'],
                mode="text",
                textposition="top center",
                hoverinfo='none'
            ))

    # Add edges for the highlighted path (in red)
    if path:
        for i in range(len(path) - 1):
            city, neighbor = path[i], path[i + 1]
            x0, y0 = coordinates[city]
            x1, y1 = coordinates[neighbor]

            # Highlight the red path
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1],
                mode="lines",
                line=dict(color="red", width=4),
                hoverinfo='none'
            ))

    # Add start and goal nodes
    fig.add_trace(go.Scatter(
        x=[coordinates[start][0]], y=[coordinates[start][1]],
        mode='markers+text',
        marker=dict(size=15, color='green'),
        text=[start], textposition="top center",
        hoverinfo='text',
        name='Start'
    ))
    fig.add_trace(go.Scatter(
        x=[coordinates[goal][0]], y=[coordinates[goal][1]],
        mode='markers+text',
        marker=dict(size=15, color='blue'),
        text=[goal], textposition="top center",
        hoverinfo='text',
        name='Goal'
    ))

    # Add other nodes (excluding start and goal)
    for city, (x, y) in coordinates.items():
        if city not in (start, goal):
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                marker=dict(size=10, color='lightblue'),
                text=[city], textposition="top center",
                hoverinfo='text',
                name=city
            ))

    # Layout settings
    fig.update_layout(
        title="Tìm đường đi ngắn nhất - Tây Nguyên",
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        showlegend=False,
        height=600,
        margin=dict(l=0, r=0, t=30, b=0)
    )

    # Display the figure in Streamlit
    st.plotly_chart(fig)

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
