# GEO5017Lab1 Linear Regression
### Group members:
1. Hongyu Ye (6286240)
2. Xinya Bi (6195350)
3. Xu Wang (6235379)

### Report
Report can be found at [report/GEO5017Lab1.pdf](https://github.com/MaxineXinyaBi/GEO5017_Lab1_LinearRegression/blob/main/Report.pdf)

### Dependence on External Libraries
- matplotlib
- numpy
- pandas
- pytorch

### The Path to Data
The data of drone location is stored in metadata.csv

### Where to Find the Results

- `1_plot_trajectory.py` is the script that plots the trajectory of the drone. When executed, it will generate Figure 1.
- `2_constant_speed_model.py` implements the constant speed model. The script employs gradient descent to optimize the parameters for each coordinate (\(x, y, z\)) and evaluates the residual errors. The tunable parameters are described in the comments. Running this script outputs the fitted equations, the number of iterations, and the residual errors.
- `3_constant_acceleration_model.py` is the implementation of the unoptimized algorithm, which includes testing the model with different learning rates. The tunable parameters are marked with comments. Running this script will output the fitting results, the number of iterations, SSE values, and generate Figure 3.
- `4_constant_acceleration_model_improved.py` is the improved version of the algorithm. There are two improvement options: one using a decaying learning rate (LR), and the other incorporating covariance and angular velocity in addition to decaying LR using PyTorch. When executed, it prints the fitting results for the `x`, `y`, and `z` coordinates, as well as the predicted values at `t = 7`, and generates the fitted function plot, e.g., Figure 7. The tunable parameters are marked with comments.

