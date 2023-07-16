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
