import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))

def add_pose(graph, initial_estimate, pose_5):
    pose_4 = initial_estimate.atPose2(X(4))
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )
    return graph, initial_estimate

def add_landmark_measurement(graph, result, pose_5, landmark):
    landmark_point = result.atPoint2(L(landmark))
    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph

def optimize(graph, initial_estimate):
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()
    return result

def minimize_marginals(graph, initial_estimate, pose_options):
    best_pose = None
    best_landmark = None
    min_sum_of_marginals = float("inf")

    for pose_key, pose_5 in pose_options.items():
        for landmark_idx in [1, 2]:
            
            # Using our clean .clone() style!
            temp_graph = graph.clone()
            temp_estimate = gtsam.Values(initial_estimate)

            temp_graph, temp_estimate = add_pose(temp_graph, temp_estimate, pose_5)
            temp_result = optimize(temp_graph, temp_estimate)

            temp_graph = add_landmark_measurement(temp_graph, temp_result, pose_5, landmark_idx)
            final_result = optimize(temp_graph, temp_estimate)

            marginals = gtsam.Marginals(temp_graph, final_result)
            total_cov = marginals.marginalCovariance(L(landmark_idx)).sum()

            if total_cov < min_sum_of_marginals:
                min_sum_of_marginals = total_cov
                best_pose = pose_key
                best_landmark = landmark_idx

    # Recalculate for the final sum
    pose_5 = pose_options[best_pose]

    temp_graph = graph.clone()
    temp_estimate = gtsam.Values(initial_estimate)

    temp_graph, temp_estimate = add_pose(temp_graph, temp_estimate, pose_5)
    temp_result = optimize(temp_graph, temp_estimate)

    temp_graph = add_landmark_measurement(temp_graph, temp_result, pose_5, best_landmark)
    final_result = optimize(temp_graph, temp_estimate)

    marginals = gtsam.Marginals(temp_graph, final_result)

    final_sum = (
        marginals.marginalCovariance(L(1)).sum() +
        marginals.marginalCovariance(L(2)).sum()
    )

    return best_pose, best_landmark, final_sum

def minimize_errors(graph, initial_estimate, pose_options):
    best_pose = None
    best_landmark = None
    min_sum_of_errors = float("inf")

    # Ground truth coordinates
    gt = {
        1: (0.0, 0.0, 0.0),
        2: (2.0, 0.0, 0.0),
        3: (4.0, 0.0, 0.0),
    }

    for pose_key, pose_5 in pose_options.items():
        for landmark_idx in [1, 2]:
            
            temp_graph = graph.clone()
            temp_estimate = gtsam.Values(initial_estimate)

            temp_graph, temp_estimate = add_pose(temp_graph, temp_estimate, pose_5)
            temp_result = optimize(temp_graph, temp_estimate)

            temp_graph = add_landmark_measurement(temp_graph, temp_result, pose_5, landmark_idx)
            final_result = optimize(temp_graph, temp_estimate)

            current_sum_of_errors = 0
            
            # Calculating absolute drift from true coordinates
            for i in [1, 2, 3]:
                pose = final_result.atPose2(X(i))
                x_gt, y_gt, theta_gt = gt[i]

                error = (
                    abs(pose.x() - x_gt) +
                    abs(pose.y() - y_gt) +
                    abs(pose.theta() - theta_gt)
                )
                current_sum_of_errors += error

            if current_sum_of_errors < min_sum_of_errors:
                min_sum_of_errors = current_sum_of_errors
                best_pose = pose_key
                best_landmark = landmark_idx

    return best_pose, best_landmark, min_sum_of_errors