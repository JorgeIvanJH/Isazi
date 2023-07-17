import numpy as np
import joblib
import networkx as nx
import ast
import pandas as pd

energy_model = joblib.load('models/energy_model.pkl')
poly_features_model = joblib.load('models/poly_features_model.pkl')

def get_energy_cost(gradient:np.array,poly_features_model,energy_model,weight:float=80,vel:float=50)->np.array:
    """
    Calculates the energy cost based on the gradient using a polynomial regression model.

    Args:
        gradient (np.array): Input gradient values.
        poly_features_model: Pre-trained polynomial features object.
        energy_model: Pre-trained regression model.
        weight [Kg]: Weight of a person
        vel [m/min]: Velocity mantained from a person

    Returns:
        np.array: Predicted energy cost values, shape (n_samples,).
    """
    gradient = gradient.reshape(-1,1)
    x_poly = poly_features_model.transform(gradient)
    y_pred = energy_model.predict(x_poly)
    energy = y_pred*weight/vel
    return energy

def slope_computing(a_h:float,b_h:float,horiz_diff:float = 10)->np.float64:
    """
    Calculates the slope between 2 different heights and a fixed horizontal displacement.

    Args:
        a_h (np.float): Actual height
        b_h (np.float): Next height
        horiz_diff (np.float): Horizontak displacement (10 meters)

    Returns:
        np.float64: slope between heights in radians
    """
    height_diff = b_h-a_h
    return np.arctan(height_diff/horiz_diff)

def build_node_graph(df_alt_map,node_graph,min_slope:float=-0.46,max_slope:float=0.46)->None:
    """
    Adds connections to a a node graph for the data presented
    Skips node connections if slope excedes the range of data provided; on which the slope model is confident
    
    Args:
        df_alt_map (np.dataframe): dataframe containing maps height
        node_graph: Pre built empty node graph
        min_slope (np.float): minimum valid slope
        max_slope (np.float): maximum valid slope

    """

    for i,row in df_alt_map.iterrows():
        
        for j,col in row.items():
            # print(f"pos: ({i},{j})")
            act_h = col
            # NORTH
            try: 
                n_h = df_alt_map.loc[i-1,j]
                slope = slope_computing(act_h,n_h)
                if min_slope<slope<max_slope:
                    energy_cost = get_energy_cost(slope,poly_features_model,energy_model)[0][0]
                    # print(f"\tNORTH: {act_h}-{n_h}") 
                    # print(f"\t\tslope: {slope}") 
                    # print(f"\t\tenergy_cost: {energy_cost}") 
                    node_graph.add_edge(f"({i},{j})",f"({i-1},{j})",weight=energy_cost)
            except: pass
            # SOUTH
            try: 
                s_h = df_alt_map.loc[i+1,j]
                slope = slope_computing(act_h,s_h)
                if min_slope<slope<max_slope:
                    energy_cost = get_energy_cost(slope,poly_features_model,energy_model)[0][0]
                    # print(f"\tSOUTH: {act_h}-{s_h}")
                    # print(f"\t\tslope: {slope}") 
                    # print(f"\t\tenergy_cost: {energy_cost}")
                    node_graph.add_edge(f"({i},{j})",f"({i+1},{j})",weight=energy_cost)
            except: pass
            # EAST
            try: 
                e_h = df_alt_map.loc[i,j+1]
                slope = slope_computing(act_h,e_h)
                if min_slope<slope<max_slope:
                    energy_cost = get_energy_cost(slope,poly_features_model,energy_model)[0][0]
                    # print(f"\tEAST: {act_h}-{e_h}")
                    # print(f"\t\tslope: {slope}") 
                    # print(f"\t\tenergy_cost: {energy_cost}") 
                    node_graph.add_edge(f"({i},{j})",f"({i},{j+1})",weight=energy_cost)
            except: pass
            # WEST
            try: 
                w_h = df_alt_map.loc[i,j-1]
                slope = slope_computing(act_h,w_h)
                if min_slope<slope<max_slope:
                    energy_cost = get_energy_cost(slope,poly_features_model,energy_model)[0][0]
                    # print(f"\tWEST: {act_h}-{w_h}")
                    # print(f"\t\tslope: {slope}") 
                    # print(f"\t\tenergy_cost: {energy_cost}") 
                    node_graph.add_edge(f"({i},{j})",f"({i},{j-1})",weight=energy_cost)
            except: pass
            # NORTH EAST
            try: 
                ne_h = df_alt_map.loc[i+1,j+1]
                slope = slope_computing(act_h,ne_h,10*2**(1/2))
                if min_slope<slope<max_slope:
                    energy_cost = get_energy_cost(slope,poly_features_model,energy_model)[0][0]
                    # print(f"\tWEST: {act_h}-{w_h}")
                    # print(f"\t\tslope: {slope}") 
                    # print(f"\t\tenergy_cost: {energy_cost}") 
                    node_graph.add_edge(f"({i},{j})",f"({i+1},{j+1})",weight=energy_cost)
            except: pass
            # SOUTH EAST
            try: 
                se_h = df_alt_map.loc[i-1,j+1]
                slope = slope_computing(act_h,w_se,10*2**(1/2))
                if min_slope<slope<max_slope:
                    energy_cost = get_energy_cost(slope,poly_features_model,energy_model)[0][0]
                    # print(f"\tWEST: {act_h}-{w_h}")
                    # print(f"\t\tslope: {slope}") 
                    # print(f"\t\tenergy_cost: {energy_cost}") 
                    node_graph.add_edge(f"({i},{j})",f"({i-1},{j+1})",weight=energy_cost)
            except: pass
            # NORTH WEST
            try: 
                nw_h = df_alt_map.loc[i+1,j-1]
                slope = slope_computing(act_h,nw_h,10*2**(1/2))
                if min_slope<slope<max_slope:
                    energy_cost = get_energy_cost(slope,poly_features_model,energy_model)[0][0]
                    # print(f"\tWEST: {act_h}-{w_h}")
                    # print(f"\t\tslope: {slope}") 
                    # print(f"\t\tenergy_cost: {energy_cost}") 
                    node_graph.add_edge(f"({i},{j})",f"({i+1},{j-1})",weight=energy_cost)
            except: pass
            # SOUTH WEST
            try: 
                sw_h = df_alt_map.loc[i-1,j-1]
                slope = slope_computing(act_h,sw_h,10*2**(1/2))
                if min_slope<slope<max_slope:
                    energy_cost = get_energy_cost(slope,poly_features_model,energy_model)[0][0]
                    # print(f"\tWEST: {act_h}-{w_h}")
                    # print(f"\t\tslope: {slope}") 
                    # print(f"\t\tenergy_cost: {energy_cost}") 
                    node_graph.add_edge(f"({i},{j})",f"({i-1},{j-1})",weight=energy_cost)
            except: pass


def compute_route(node_graph,starting_point:tuple=(0,0),end_point:tuple=(200,559))->pd.DataFrame:
    """
    Calulates the trail path which minimixes the energy excerted
    
    Args:
        node_graph: node graph with the needed connections
        starting_point (tuple): wanted starting point (x,y)
        end_point (tuple):  wanted ending point (x,y)
    Returns:
        pd.DataFrame: slope between heights in radians
    """
    starting_point = starting_point[::-1]
    end_point = end_point[::-1]
    starting_point = f"({starting_point[0]},{starting_point[1]})"
    end_point = f"({end_point[0]},{end_point[1]})"
    route_str = nx.shortest_path(node_graph, starting_point, end_point, weight='weight')
    route = [ast.literal_eval(elem) for elem in route_str]
    route_df = pd.DataFrame(route,columns=['y_coord','x_coord'])
    return route_df