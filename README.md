# Beginner's Trail Planning

## Author
**Name:** Jorge Iván Jaramillo Herrera  
**LinkedIn:** [Jorge Iván Jaramillo Herrera](https://www.linkedin.com/in/jorge-iv%C3%A1n-jaramillo-herrera-1a3849117/)

## Instructions
To run the trail planning script, execute the following command in this directory:
```bash
python plan_trail.py
```

## Directory Structure
- **models:** Directory where all developed models are saved and imported from to reduce computing time.
- **data:** Directory where the provided data is stored and read from.
- **planned_routes:** Directory where results (.png and .csv) are saved.

## Scripts and Text Files
- **endurance_trail_proposal.txt:** Ideas for a future endurance trail planning algorithm.
- **test.ipynb:** Jupyter notebook containing the entire problem-solving process: Ingestion, Modelling, Optimisation, and Simple Reporting.
- **functions.py:** Contains all functions needed for the computations.
- **plan_trail.py:** The final executable code. Run with:
    ```bash
    python plan_trail.py
    ```
    You will be asked to select a value between 0 and 682 corresponding to the starting point of the route.

## Project Overview
The goal is to optimize the path of a new trail in a mountainous region of the Free State. The trail is intended to be a beginner’s trail that does not require too much exertion from its participants.

### Provided Data
- **Altitude Map:** A .csv file describing the altitude map of the region.
- **Energy Expenditure Measurements:** A .csv file from a sports science lab relating walking gradient/slope on a treadmill with the energy expended by multiple test subjects.

### Solution Components
1. **Ingestion**
    - Read from the .csv files and extract relevant information.
    - Note: The altitude map has a resolution of 10m x 10m.
    - Note: Energy expenditure is measured in J.kg-1.min-1.
    - Note: Altitude map measurements are in meters, with North and the Y-axis going up vertically.

2. **Modelling**
    - Use applicable statistics/machine learning methods to predict a person’s expected energy expenditure for a given gradient.

3. **Optimisation**
    - Find a path from any point on the Southern border of the map to a lodge entrance at x=200 and y=559, minimizing total expected exertion (in Joules).
    - Use any optimization method that runs in a reasonable amount of time (less than 10 mins on standard hardware).
    - Assume trail participants have a fixed body mass and a fixed walking speed.

4. **Simple Reporting**
    - Write the path solution to a .csv file with columns: x_coord and y_coord.
    - Write the path, overlaid on the altitude map, to a .png file.
    - Solution coordinates should be measured from the South-Western corner of the map (x=0, y=0) and correspond to the resolution of the altitude map (one point for every 10m² square).
    - Provide advice for a future endurance trail in a .txt file, explaining what additional information is needed and how the approach might change.

The above steps should run end-to-end (.csv to results) with a single call to a script. The solution must use only open-source tools (e.g., Python, R). Specify any external libraries or packages used in the README file.
