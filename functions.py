import numpy as np
import joblib


energy_model = joblib.load('models/energy_model.pkl')
poly_features_model = joblib.load('models/poly_features_model.pkl')

def get_energy_cost(gradient:np.array,poly_features_model,energy_model)->np.array:
    """
    Calculates the energy cost based on the gradient using a polynomial regression model.

    Args:
        gradient (np.array): Input gradient values.
        poly_features_model: Pre-trained polynomial features object.
        energy_model: Pre-trained regression model.

    Returns:
        np.array: Predicted energy cost values, shape (n_samples,).
    """
    gradient = gradient.reshape(-1,1)
    x_poly = poly_features_model.transform(gradient)
    y_pred = energy_model.predict(x_poly)
    return y_pred

def slope_computing(a_h:float,b_h:float,horiz_diff:float = 10):
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

def build_node_graph(df_alt_map,node_graph,min_slope:float=-0.46,max_slope:float=0.46):
    """
    Skips node connections if slope excedes the range of data provided and on which the slope model is confident
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