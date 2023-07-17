import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import networkx as nx
import ast
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from functions import get_energy_cost,energy_model, poly_features_model, slope_computing,build_node_graph,compute_route


# READ MAP
df_alt_map = pd.read_csv('data/altitude_map (1).csv',header=None) # reading map info (altitude in [m])
df_alt_map.index = df_alt_map.index[::-1] # South-Western corner of the map (x=0, y=0)
# READ OR BUILD/SAVE NODE GRAPH IF NOT FOUND
try:
    node_graph = pickle.load(open('models/node_graph.pickle', 'rb'))
except:
    node_graph=nx.Graph()
    build_node_graph(df_alt_map,node_graph)
    pickle.dump(node_graph, open('models/node_graph.pickle', 'wb'))

max_pos_idx = df_alt_map.index[-1]
min_pos_col = df_alt_map.columns[0]
max_pos_col = df_alt_map.columns[-1]
horiz_select = -1
print("Please select trails starting position\n\nNotes:\n")
print("\t- Trail starts anywhere on mountains south-most position")
print("\t- Fixed destination: Lodge located on (200,559)\n")
# print(f"Enter an value between: {min_pos_col} and {max_pos_col} and then hit Enter\n")
while horiz_select not in  df_alt_map.columns:
    horiz_select = input(f"Enter an value between: {min_pos_col} and {max_pos_col} and then hit Enter: ")
    horiz_select = int(horiz_select)

starting_point = (horiz_select,max_pos_idx)
print(f"Route will start on x = {starting_point[0]}, y = {starting_point[1]}")
print(f"Route will end on x = 200, y = 559")
route_df = compute_route(node_graph,starting_point)
fig, ax = plt.subplots(figsize=(10, 10))
df_alt_map_aux = df_alt_map.copy()
df_alt_map_aux.index = [f"{elem*10}-{elem*10+10}" for elem in df_alt_map_aux.index] # set y_orig axis resolution (10m)
df_alt_map_aux.columns = [f"{elem*10}-{elem*10+10}" for elem in df_alt_map_aux.columns] # set x axis resolution (10m)
sns.heatmap(df_alt_map_aux, ax=ax)
route_df.y_coord = route_df.y_coord[::-1].values # fix scatterplot axis
sns.scatterplot(data=route_df, x="x_coord", y="y_coord", ax=ax)
# plt.figure(facecolor='white')
ax.set(xlabel='West - East [m]', ylabel='South - North [m]',title='Altitude Map')
# SAVE RESULTS
route_df.to_csv(f'planned_routes/route_x_{starting_point[0]}_y_{starting_point[1]}.csv', index = False, encoding='utf-8')
plt.savefig(f'planned_routes/route_x_{starting_point[0]}_y_{starting_point[1]}.png', facecolor='white')
print("Please check planned_routes folder for paths csv and png files ☺")