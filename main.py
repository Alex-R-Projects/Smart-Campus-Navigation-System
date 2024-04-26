
import matplotlib.pyplot as plt # For Visualizations
import osmnx as ox


# For this project we are going to use the google maps API for be able to 1) Visualize, and 2) use as a graph that way we can implement the many shortest path algorithms 

# Coordinates for CSUF
latitude = 33.8823
longitude = -117.8851

# Retrieve the graph of the university campus
# dist = distance in meters, we can change the parameter to adjust how detailed we want the graph to be 
csuf_campus_map = ox.graph_from_point((latitude, longitude), dist=800, network_type='all')
# Network_type = pathways, like bike ways, streets, etc

# List of locations within Cal State Fullerton
csuf_locations = { # dictionary to hold the location names, and the coordinates associated with them
    "Humanities Building": (33.8836, -117.8836),
    "McCarthy Hall": (33.8819, -117.8849),
    "Pollak Library": (33.8827, -117.8845),
    "Titan Student Union": (33.8824, -117.8856),
    "College Park Building": (33.8840, -117.8847),
    "Langsdorf Hall": (33.8823, -117.8859),
    "Mihaylo College of Business and Economics": (33.8832, -117.8865),
    "Steven G. Mihaylo Hall": (33.8831, -117.8867),
    "Gordon Hall": (33.8835, -117.8852),
    "Student Recreation Center": (33.8847, -117.8842),
    "Parking Structure": (33.8820, -117.8844),
    "Titan Gym": (33.8843, -117.8840),
    "Engineering Building": (33.8830, -117.8861),
    "Visual Arts Center": (33.8828, -117.8870),
    "Performing Arts Center": (33.8843, -117.8847),
    "Nutwood Parking Structure": (33.8852, -117.8847),
    "Athletics Fields": (33.8834, -117.8825),
    "Education Classroom Building": (33.8833, -117.8874),
    "Eastside Parking Structure": (33.8815, -117.8812),
    "State College Parking Structure": (33.8848, -117.8883)
}

# Add locations as nodes to the graph
for location, coords in csuf_locations.items():
    csuf_campus_map.add_node(location, x = coords[1], y = coords[0])

# Plot the updated graph to visualize it
ox.plot_graph(csuf_campus_map, node_size=0, show=False, close=False)

# Plot location nodes on top of the campus map
for node in csuf_campus_map.nodes(data=True):
    # Note: coordinates are accessed as (latitude, longitude)
    # Nodes will be represented with red circles
    if node[0] in csuf_locations:
        pos = node[1]['y'], node[1]['x']  
        plt.plot(pos[1], pos[0], 'ro', markersize=5)  


# Display the plot
plt.title("Cal State Fullerton Campus Map with Nodes")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
