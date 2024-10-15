import tkinter as tk
from tkinter import messagebox
from tkinter import ttk  # Use ttk for modern widgets
import matplotlib.pyplot as plt
import networkx as nx
from simpleai.search import  breadth_first, SearchProblem
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

    # If there's a valid result, extract the path and add the start city at the beginning
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

# Draw the map and highlight the path, marking start in green and goal in blue
def draw_map(path, start, goal):
    G = nx.Graph()
    for city, neighbors in graph.items():
        for neighbor, distance in neighbors.items():
            G.add_edge(city, neighbor, weight=distance)

    pos = coordinates
    plt.figure(figsize=(8, 6))  # Adjusted the size to make it smaller and fit better

    # Draw the graph with all paths
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1000, font_size=7, font_weight='semibold')

    if path:
        path_edges = list(zip(path[:-1], path[1:]))  # Create the list of edges in the path
        # Draw the shortest path found
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=3)  # Highlight path in red
        # Highlight the nodes in the path (excluding start and goal)
        nx.draw_networkx_nodes(G, pos, nodelist=path[1:-1], node_color='red', node_size=1000)

    # Highlight the start and goal nodes
    nx.draw_networkx_nodes(G, pos, nodelist=[start], node_color='green', node_size=1200)  # Start in green
    nx.draw_networkx_nodes(G, pos, nodelist=[goal], node_color='blue', node_size=1200)  # Goal in blue

    # Draw distances for all edges on the map
    for city, neighbors in graph.items():
        for neighbor, distance in neighbors.items():
            # Calculate the position to place the distance label
            mid_x = (pos[city][0] + pos[neighbor][0]) / 2
            mid_y = (pos[city][1] + pos[neighbor][1]) / 2
            
            # Draw the distance label
            plt.text(mid_x, mid_y, f"{distance} km", fontsize=8, ha='center', color='black')

    plt.title("Bản đồ đường đi Tây Nguyên")
    plt.show()

# Tkinter window setup
root = tk.Tk()
root.title("Tìm đường đi ngắn nhất - Tây Nguyên")

window_width = 400
window_height = 400
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width // 2) - (window_width // 2)
y = (screen_height // 2) - (window_height // 2)
root.geometry(f"{window_width}x{window_height}+{x}+{y}")
root.resizable(False, False)

cities = list(graph.keys())

algorithm_var = tk.StringVar(value='simpleai')

def on_search():
    start = start_var.get()
    goal = goal_var.get()
    algorithm = algorithm_var.get()
    
    if start == goal:
        messagebox.showinfo("Thông báo", "Điểm xuất phát và điểm đến phải khác nhau!")
        return

    if algorithm == 'simpleai':
        solution = find_shortest_path_simpleai(start, goal)
        if solution:  # Check if a solution was found
            path = [node[1] for node in solution.path()]  # Extract only the destination city from each tuple
        else:
            path = []  # No path found
    else:
        solution = find_shortest_path_aima(start, goal)
        path = solution if solution else []  # In AIMA, the result is already a list of the path

    if path:
        # Calculate and display the total distance
        total_distance = calculate_total_distance(path)

        # Print the path and distance in a messagebox
        shortest_path = ' -> '.join(path)
        messagebox.showinfo("Kết quả", f"Đường đi ngắn nhất: {shortest_path}\nTổng quãng đường: {total_distance} km")
        
        # Draw the map with the shortest path highlighted
        draw_map(path, start, goal)
    else:
        messagebox.showinfo("Thông báo", "Không tìm thấy đường đi ngắn nhất giữa hai điểm này.")

# Tkinter Widgets
header_label = ttk.Label(root, text="Tìm đường đi ngắn nhất", font=("Helvetica", 16))
header_label.pack(pady=10)

start_label = ttk.Label(root, text="Chọn điểm xuất phát:")
start_label.pack(anchor='w', padx=20)

start_var = tk.StringVar(value=cities[0])
start_menu = ttk.Combobox(root, textvariable=start_var, values=cities)
start_menu.pack(pady=5)

goal_label = ttk.Label(root, text="Chọn điểm đến:")
goal_label.pack(anchor='w', padx=20)

goal_var = tk.StringVar(value=cities[1])
goal_menu = ttk.Combobox(root, textvariable=goal_var, values=cities)
goal_menu.pack(pady=5)

algorithm_label = ttk.Label(root, text="Chọn thuật toán tìm kiếm:")
algorithm_label.pack(anchor='w', padx=20)

algorithm_frame = ttk.Frame(root)
algorithm_frame.pack(pady=5)

simpleai_radio = ttk.Radiobutton(algorithm_frame, text="SimpleAI (BFS)", variable=algorithm_var, value='simpleai')
simpleai_radio.pack(side='left', padx=10)

aima_radio = ttk.Radiobutton(algorithm_frame, text="AIMA (A*)", variable=algorithm_var, value='aima')
aima_radio.pack(side='left')

search_button = ttk.Button(root, text="Tìm kiếm", command=on_search)
search_button.pack(pady=20)

# Display student info
info_project = ttk.Label(root, text="Đồ án môn học: Trí tuệ nhân tạo\n" + \
    "Tên đề tài: Tìm đường đi ngắn nhất giữa các\n huyện và thành phố của các tỉnh Tây Nguyên"
)
info_project.pack(anchor='w', padx=20)

info_label = ttk.Label(root, text="Sinh viên: Vũ Xuân Quang\nMSSV: 22110212")
info_label.pack(anchor='w', padx=20)


root.mainloop()
