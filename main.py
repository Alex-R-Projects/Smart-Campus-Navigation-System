from gmap_api_key import *
import matplotlib as plt # For Visualizations
import osmnx as ox
# For this project we are going to use the google maps API for be able to 1) Visualize, and 2) use as a graph that way we can implement the many shortest path algorithms 

# Coordinates for CSUF
latitude = 33.8823
longitude = -117.8851




# Retrieve the graph of the university campus
# dist = distance in meters, we can change the parameter to adjust how detailed we want the graph to be 
csuf_campus_map = ox.graph_from_point((latitude, longitude), dist=1000, network_type='all')
# Network_type = pathways, like bike ways, streets, etc

# Plotting the graph to visualize it
ox.plot_graph(csuf_campus_map, node_size=0)

