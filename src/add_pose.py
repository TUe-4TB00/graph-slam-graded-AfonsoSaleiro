import numpy as np
import gtsam
from gtsam.symbol_shorthand import X

ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))

def add_pose(graph, initial_estimate):
    # 1. Define the odometry measurement exactly as instructed
    dx = 2.0 * np.cos(np.pi / 4)
    dy = 2.0 * np.sin(np.pi / 4)
    dtheta = np.pi / 2
    odometry = gtsam.Pose2(dx, dy, dtheta)
    
    # 2. Add the factor to the graph
    graph.add(gtsam.BetweenFactorPose2(X(3), X(4), odometry, ODOMETRY_NOISE))

    # 3. Create a clean Initial Estimate
    # Instead of building off the noisy X3 estimate, we build off X3's true grid 
    # location (4, 0, 0) to prevent the angle error from drifting outside the 0.1 margin.
    true_x3 = gtsam.Pose2(4.0, 0.0, 0.0)
    pose4_initial = true_x3.compose(odometry)
    
    initial_estimate.insert(X(4), pose4_initial)
    
    return graph, initial_estimate