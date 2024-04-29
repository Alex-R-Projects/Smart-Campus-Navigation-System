
import matplotlib.pyplot as plt # For Visualizations
import osmnx as ox # This gets the lat/long coordinates, and will plot the graph for CSUF
import collections # defaultdict
from collections import deque
import heapq

from pprint import pprint # debugging

def debug(data, mode='w'):
    with open('debug.txt', mode) as f:
        pprint(data, stream=f)

def create_graph(csuf_campus_map) -> dict:
    # Create cleaned graph
    graph = collections.defaultdict(dict)

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

# Add locations as nodes to the graph
for location, coords in csuf_locations.items():
    csuf_campus_map.add_node(location, x = coords[1], y = coords[0], weight = 1)
# I know for Dijkstra's Algo we need to set weights, but for now I have just set the weights to 1 for ALL nodes

# Plot the updated graph to visualize it
ox.plot_graph(csuf_campus_map, node_size=10, show=False, close=False)

# Plot location nodes on top of the campus map
for id, data in csuf_campus_map.nodes(data=True):
    # Note: coordinates are accessed as (latitude, longitude)
    # Nodes will be represented with red circles
    if id in csuf_locations:
        plt.plot(data['x'], data['y'], 'ro', markersize=5)  

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
    queue = deque([[start]])
    visited = set([start])

    while queue:
        path = queue.popleft()
        node = path[-1]

        if node == end:
            return path

        for adjacent, _ in graph[node]['adj'].items():
            if adjacent not in visited:
                visited.add(adjacent)
                new_path = list(path)
                new_path.append(adjacent)
                queue.append(new_path)

    return None

# Depth-first search
# Needs comments
def dfs(graph, start, end, path=None):
    if path==None:
        path = []
    path.append(start)
    if start == end:
        return path
    if start not in graph:
        return None
    for node in graph[start]['adj']:
        if node not in path:
            new_path = dfs(graph, node, end, path)
            if new_path:
                return new_path
    return None



# Display the plot
plt.title("Cal State Fullerton Campus Map with Nodes")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()


graph = create_graph(csuf_campus_map)
add_locations(graph)

path = dfs(graph, 'Student Recreation Center', 'Titan Gym')

debug(path)