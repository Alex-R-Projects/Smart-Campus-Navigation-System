import time
import heapq
from collections import deque, defaultdict

import matplotlib.pyplot as plt # For Visualizations
from matplotlib.widgets import Button, TextBox
import osmnx as ox # This gets the lat/long coordinates, and will plot the graph for CSUF

from pprint import pprint # debugging

def debug(data, mode='w'):
    with open('debug.txt', mode) as f:
        pprint(data, stream=f)

def create_graph(csuf_campus_map) -> dict:
    # Create cleaned graph
    graph = defaultdict(dict)

    # Copy adjacency
    for source_id, source_data in csuf_campus_map.adj.items():
        for nei_id, nei_data in source_data.items():
            # Not sure what the 0 key is for
            # print(csuf_campus_map.adj)
            nei_data = nei_data[0]
            
            if 'name' in nei_data:
                graph[nei_id]['name'] = nei_data['name']
            if 'adj' not in graph[source_id]:
                graph[source_id]['adj'] = {}
            if 'adj' not in graph[nei_id]:
                graph[nei_id]['adj'] = {}
            graph[source_id]['adj'][nei_id] = nei_data['length']
    
    # Copy coords
    for name, data in csuf_campus_map.nodes(data=True):
        long, lat = data['x'], data['y']
        graph[name]['coords'] = (lat, long)

    return graph


def add_locations(graph) -> None:
    # The OSMNX library only stores street names, not our locations
    # Add our locations to the graph by connecting each location to its nearest street(s)
    for location_name, (lat, long) in csuf_locations.items():
        graph[location_name]['name'] = location_name
        graph[location_name]['adj'] = {}
        graph[location_name]['coords'] = (lat, long)

    for location_name, (lat, long) in csuf_locations.items():
        # Bounds padding
        # Create a square around the building to connect more possible streets
        # Arbitrary value (for now)
    # TESTING REQUIRED
        bp = 0.0001 # unit: long/lat
        top_left = (-bp, -bp)
        top_right = (bp, -bp)
        bottom_left = (-bp, bp)
        bottom_right = (bp, bp)
        for dx, dy in [top_left, top_right, bottom_left, bottom_right]:

            nearest_node_id, dist = ox.distance.nearest_nodes(csuf_campus_map, long + dx, lat + dy, return_dist=True)

            scale = 1
            while nearest_node_id == location_name:
                scale *= 1.4
                nearest_node_id, dist = ox.distance.nearest_nodes(csuf_campus_map, long + dx * scale, lat + dy * scale, return_dist=True)

            graph[location_name]['adj'][nearest_node_id] = dist
            graph[nearest_node_id]['adj'][location_name] = dist

# Coordinates for CSUF
latitude = 33.8823
longitude = -117.8851

# Retrieve the graph of the university campus
# dist = distance in meters, we can change the parameter to adjust how detailed we want the graph to be 
csuf_campus_map = ox.graph_from_point((latitude, longitude), dist=800, network_type='all')
# Network_type = pathways, like bike ways, streets, etc

csuf_locations = { # dictionary to hold the location names, and the coordinates associated with them
    "Humanities Building":                       (33.880501635139034, -117.88410082595438),
    "McCarthy Hall":                             (33.879645105130535, -117.8855720282093),
    "Pollak Library":                            (33.881226230203225, -117.88537890928674),
    "Titan Student Union":                       (33.881789266771854, -117.88848473075156),
    "College Park Building":                     (33.87757498322724, -117.8834483491749),
    "Langsdorf Hall":                            (33.879007227971314, -117.88434605269099),
    "Mihaylo College of Business and Economics": (33.87876744962989, -117.88331719822932),
    "Steven G. Mihaylo Hall":                    (33.878737420937604, -117.88330674089343),
    "Gordon Hall":                               (33.87974056822556, -117.88416485364138),
    "Student Recreation Center":                 (33.883151509614216, -117.88777544125246),
    "Parking Structure":                         (33.8820, -117.8844), # which parking structure is this?
    "Titan Gym":                                 (33.88311484580521, -117.88601834470772),
    "Engineering Building":                      (33.88223141593597, -117.88278388630202),
    "Visual Arts Center":                        (33.88007014225456, -117.88866610469454),
    "Performing Arts Center":                    (33.8804928643185, -117.8866998236957),
    "Nutwood Parking Structure":                 (33.87904780275712, -117.88854300210211),
    "Athletics Fields":                          (33.883885019638015, -117.8856008766424),
    "Education Classroom Building":              (33.881271257706345, -117.88434071135715),
    "Eastside Parking Structure":                (33.88023716960995, -117.8817502188979),
    "State College Parking Structure":           (33.883469482027344, -117.88868179746281)
}



# Dijkstra's Algorithm
def dijkstra(graph, start, end):
    # Initialize the distance dictionary with infinite distances for all nodes except the start node
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0

    # Initialize the path dictionary
    paths = {node: [] for node in graph}
    paths[start] = [start]

    # Priority queue for the nodes to visit
    queue = [(0, start)]

    while queue:
        # Get the node with the smallest distance
        current_distance, current_node = heapq.heappop(queue)

        # If the current distance is greater than the recorded distance for the current node, skip it
        if current_distance > distances[current_node]:
            continue

        # Check all the neighboring nodes
        for neighbor, weight in graph[current_node]['adj'].items():
            distance = current_distance + weight

            # If the calculated distance is less than the recorded distance for the neighbor, update it
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                paths[neighbor] = paths[current_node] + [neighbor]
                heapq.heappush(queue, (distance, neighbor))

    # Return the shortest path from start to end
    return paths[end]

# Breadth-first search
# Needs comments
def bfs(graph, start, end):
    queue = [(0, start, [])]
    visited = set()

    while queue:
        (dist, node, path) = heapq.heappop(queue)
        if node not in visited:
            visited.add(node)
            path = path + [node]
            if node == end:
                return path
            for adjacent, weight in graph[node]['adj'].items():
                if adjacent not in visited:
                    heapq.heappush(queue, (dist + weight, adjacent, path))

    return None

# Depth-first search
# This shit is broken
def dfs(graph, start, end, visited=None, path=None):
    if visited is None:
        visited = set()
    if path is None:
        path = [start]

    if start == end:
        return path

    visited.add(start)

    for node in graph[start]['adj']:
        if node not in visited:
            new_path = dfs(graph, node, end, visited, path + [node])
            if new_path:
                return new_path

    return None


################## Code for Plot is below ####################
def plot_exec_time(start_t, end_t):
    plt.text(0.0, 0.01, f"Execution time: {end_t-start_t}", color='red', fontsize=12, transform=plt.gca().transAxes)

def err_invalid():
    start_textbox.set_val("Invalid")
    end_textbox.set_val("Invalid")

def run_bfs(graph):
    def run(event):
        start = start_textbox.text
        end = end_textbox.text

        if start not in graph or end not in graph:
            err_invalid()
            return

        s_t = time.perf_counter()
        path = bfs(graph, start, end)
        e_t = time.perf_counter()
        plot_path(graph, path)
        plot_exec_time(s_t, e_t)
        plt.show()
        debug(f"bfs from {start} to {end}: {path}")
    return run

# Function to handle button click for DFS 
def run_dfs(graph):
    def run(event):
        start = start_textbox.text
        end = end_textbox.text

        if start not in graph or end not in graph:
            err_invalid()
            return
        
        s_t = time.perf_counter()
        path = dfs(graph, start, end)
        e_t = time.perf_counter()
        plot_path(graph, path)
        plot_exec_time(s_t, e_t)
        plt.show()
        debug(f"dfs from {start} to {end}: {path}")
    return run

# Function to handle button click for Dijkstra's
def run_dijkstra(graph):
    def run(event):
        start = start_textbox.text
        end = end_textbox.text

        if start not in graph or end not in graph:
            err_invalid()
            return

        s_t = time.perf_counter()
        path = dijkstra(graph, start, end)
        e_t = time.perf_counter()
        plot_path(graph, path)
        plot_exec_time(s_t, e_t)
        plt.show()
        debug(f"dijkstra from {start} to {end}: {path}")
    return run

def clear_textboxes(event):
    start_textbox.set_val('')
    end_textbox.set_val('')

# Add locations as nodes to the graph
for location, coords in csuf_locations.items():
    csuf_campus_map.add_node(location, x = coords[1], y = coords[0], weight = 1)
# I know for Dijkstra's Algo we need to set weights, but for now I have just set the weights to 1 for ALL nodes

def plot_map():
    # Plot the updated graph to visualize it
    ox.plot_graph(csuf_campus_map, node_size=10, show=False, close=False)
    # Plot location nodes on top of the campus map
    for id, data in csuf_campus_map.nodes(data=True):
        # Note: coordinates are accessed as (latitude, longitude)
        # Nodes will be represented with red circles
        if id in csuf_locations:
            plt.plot(data['x'], data['y'], 'ro', markersize=5)  
    # Display the plot
    plt.title("Cal State Fullerton Campus Map with Nodes")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

def plot_path(graph, path):
    plot_map()

    start_y, start_x = graph[path[0]]['coords']
    end_y, end_x = graph[path[-1]]['coords']
    coords_x = []
    coords_y = []
    for node in path:
        y, x = graph[node]['coords']
        coords_x.append(x)
        coords_y.append(y)

    plt.plot(coords_x, coords_y, '.y-', lw=2)
    plt.plot(start_x, start_y, 'go', markersize=10)
    plt.plot(end_x, end_y, 'bo', markersize=10)

    plt.annotate('Start', (start_x, start_y), textcoords="offset points", xytext=(-20,-20), ha='center', color='black', bbox=dict(boxstyle="round,pad=0.3", fc="white", lw=2))
    plt.annotate('End', (end_x, end_y), textcoords="offset points", xytext=(-20,-20), ha='center', color='black', bbox=dict(boxstyle="round,pad=0.3", fc="white", lw=2))

def plot_location_names(event):
    plt.figure()
    plt.title('')
    
    text = []
    for location in csuf_locations.keys():
        text.append(f"{location}\n")
    
    plt.text(0.3, 0.1, ''.join(text), fontsize=10, color='black', transform=plt.gca().transAxes)
    plt.axis('off')
    plt.show()

plot_map()

graph = create_graph(csuf_campus_map)
add_locations(graph)

 # Below are the dimensions for buttons for the each algo
bfs_ax = plt.axes([0.625, 0.005, 0.11, 0.1])
dfs_ax = plt.axes([0.75, 0.005, 0.11, 0.1])
dijkstra_ax = plt.axes([0.875, 0.005, 0.11, 0.1])
# Create text input boxes for start and end points
start_ax = plt.axes([0.075, 0.05, 0.2, 0.04])
end_ax = plt.axes([0.35, 0.05, 0.2, 0.04])
clear_ax = plt.axes([0.075, 0.01, 0.2, 0.03])
show_locations_ax = plt.axes([0.35, 0.01, 0.2, 0.03])

start_textbox = TextBox(start_ax, 'Start:')
end_textbox = TextBox(end_ax, 'End:')


bfs_button = Button(bfs_ax, 'Run BFS')
bfs_button.on_clicked(run_bfs(graph))

dfs_button = Button(dfs_ax, 'Run DFS')
dfs_button.on_clicked(run_dfs(graph)) # 

dijkstra_button = Button(dijkstra_ax, 'Run Dijkstras')
dijkstra_button.on_clicked(run_dijkstra(graph))

clear_button = Button(clear_ax, 'Clear')
clear_button.on_clicked(clear_textboxes)

show_locations_button = Button(show_locations_ax, 'Location Names')
show_locations_button.on_clicked(plot_location_names)

plt.show()

path = dijkstra(graph, 'College Park Building', 'Titan Gym')

debug(path)