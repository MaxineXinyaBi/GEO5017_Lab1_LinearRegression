import numpy as np

# for constant speed model, the formula to fit the points would be pos = initial_pos + speed * time
# for acceleration speed model, the formula to fit would be pos = initial_pos + speed * time + acceleration * time ^2
def gradient_descent(start, gradient_func, learning_rate, max_iterations, threshold):
    """
       Implementation of gradient descent algorithm

       Parameters:
       start: Initial guess for parameters, constant speed model would be [0,1], acceleration model would be [0,1,1]
       gradient_func: Function that returns gradient at current parameters
       learning_rate: Step size for parameter updates
       max_iterations: Maximum number of iterations
       threshold: Stops when step size is below this value

       Returns:
       params: Optimized parameters
       """
    params = np.array(start)
    for i in range(max_iterations):
        # calculate the step size
        step_size = learning_rate * gradient_func(params)
        # if step size is too small, quit the function
        if np.all(np.abs(step_size) < threshold):
            print(f"stop after {max_iterations} iterations, current parameters: {params}")
            break
        # calculate the new params
        params = params - step_size
    print(f"Optimized parameters: {params}")
    return params
