"""
Initially written by Ming Hsiao in MATLAB
Rewritten in Python by Wei Dong (weidong@andrew.cmu.edu), 2021
"""

import time
import numpy as np
import scipy.linalg
from scipy.sparse import csr_matrix
import tyro
from dataclasses import dataclass, field
from typing import Literal
import matplotlib.pyplot as plt
from solvers import *
from utils import *


def create_linear_system(
    odoms, observations, sigma_odom, sigma_observation, n_poses, n_landmarks
):
    r"""
    \param odoms Odometry measurements between i and i+1 in the global coordinate system. Shape: (n_odom, 2).
    \param observations Landmark measurements between pose i and landmark j in the global coordinate system. Shape: (n_obs, 4).
    \param sigma_odom Shared covariance matrix of odometry measurements. Shape: (2, 2).
    \param sigma_observation Shared covariance matrix of landmark measurements. Shape: (2, 2).

    \return A (M, N) Jacobian matrix.
    \return b (M, ) Residual vector.
    where M = (n_odom + 1) * 2 + n_obs * 2, total rows of measurements.
          N = n_poses * 2 + n_landmarks * 2, length of the state vector.
    """

    n_odom = len(odoms)
    n_obs = len(observations)

    M = (n_odom + 1) * 2 + n_obs * 2
    N = n_poses * 2 + n_landmarks * 2

    A = np.zeros((M, N))
    b = np.zeros((M,))

    # Prepare Sigma^{-1/2}.
    sqrt_inv_odom = np.linalg.inv(scipy.linalg.sqrtm(sigma_odom))
    sqrt_inv_obs = np.linalg.inv(scipy.linalg.sqrtm(sigma_observation))

    # TODO: First fill in the prior to anchor the 1st pose at (0, 0)
    sigma_p = np.diag([0.1, 0.1])
    sqrt_inv_p = np.linalg.inv(scipy.linalg.sqrtm(sigma_p))
    # The jacobian of r0 wrt r0 is identity and 0 otherwise
    A[:2, :2] = sqrt_inv_p @ np.identity(2)

    # TODO: Then fill in odometry measurements
    J_odom = np.array([[-1, 0, 1, 0], [0, -1, 0, 1]])
    A_odom = sqrt_inv_odom @ J_odom
    for i in range(n_odom):
        A[2 + 2 * i : 2 + 2 * i + 2, 2 * i : 2 * i + 4] = A_odom

    # TODO: Then fill in landmark measurements
    J_obs = np.array([[-1, 0, 1, 0], [0, -1, 0, 1]])
    A_obs = sqrt_inv_obs @ J_obs
    A_obs_pose = A_obs[:, :2]
    A_obs_landmark = A_obs[:, 2:]
    for i in range(n_obs):
        pose_idx = int(observations[i, 0])
        landmark_idx = int(observations[i, 1])
        pose_col = 2 * pose_idx
        landmark_col = 2 * n_poses + 2 * landmark_idx
        obs_row = 2 + 2 * n_odom + 2 * i

        A[obs_row : obs_row + 2, pose_col : pose_col + 2] = A_obs_pose
        A[obs_row : obs_row + 2, landmark_col : landmark_col + 2] = A_obs_landmark

    # Construct b
    clarity: b[0:2] = (sqrt_inv_p @ np.array([0, 0])).flatten

    for i in range(n_odom):
        b[2 + 2 * i : 2 + 2 * i + 2] = (
            sqrt_inv_odom @ odoms[i, :].reshape(-1, 1)
        ).flatten()

    for i in range(n_obs):
        b[2 + 2 * n_odom + 2 * i : 2 + 2 * n_odom + 2 * i + 2] = (
            sqrt_inv_obs @ observations[i, 2:].reshape(-1, 1)
        ).flatten()

    return csr_matrix(A), b


@dataclass
class Args:
    data: str = "../data/2d_linear.npz"
    """Path to npz file."""
    method: list[
        Literal["default", "pinv", "qr", "lu", "qr_colamd", "lu_colamd"]
    ] = field(default_factory=lambda: ["default"])
    repeats: int = 1
    """Number of repeats in evaluation efficiency. Increase to ensure stability."""


if __name__ == "__main__":
    args = tyro.cli(Args)

    data = np.load(args.data)

    # Plot gt trajectory and landmarks for a sanity check.
    gt_traj = data["gt_traj"]
    gt_landmarks = data["gt_landmarks"]
    plt.plot(gt_traj[:, 0], gt_traj[:, 1], "b-", label="gt trajectory")
    plt.scatter(
        gt_landmarks[:, 0], gt_landmarks[:, 1], c="b", marker="+", label="gt landmarks"
    )
    plt.legend()
    plt.show()

    n_poses = len(gt_traj)
    n_landmarks = len(gt_landmarks)

    odoms = data["odom"]
    observations = data["observations"]
    sigma_odom = data["sigma_odom"]
    sigma_landmark = data["sigma_landmark"]

    # Build a linear system
    A, b = create_linear_system(
        odoms, observations, sigma_odom, sigma_landmark, n_poses, n_landmarks
    )

    # Solve with the selected method
    for method in args.method:
        print(f"Applying {method}")

        total_time = 0
        total_iters = args.repeats
        for i in range(total_iters):
            start = time.time()
            x, R = solve(A, b, method)
            end = time.time()
            total_time += end - start
        print(f"{method} takes {total_time / total_iters}s on average")

        if R is not None:
            plt.spy(R)
            plt.show()

        traj, landmarks = devectorize_state(x, n_poses)

        # Visualize the final result
        plot_traj_and_landmarks(traj, landmarks, gt_traj, gt_landmarks)
