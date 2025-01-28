import cv2
import numpy as np

class CameraPoseEstimator:
    def __init__(self, camera_matrix=None, dist_coeffs=None):
        """
        Initialize the CameraPoseEstimator with camera intrinsic parameters.

        Args:
            camera_matrix (np.ndarray): 3x3 camera intrinsic matrix.
            dist_coeffs (np.ndarray, optional): Distortion coefficients. Defaults to None.
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros((4, 1))
    

    def estimate_pose(self, points_3d, points_2d, method=cv2.SOLVEPNP_ITERATIVE):
        """
        Estimate the camera pose from 3D-2D point correspondences.

        Args:
            points_3d (np.ndarray): Array of 3D points in the world coordinate system (Nx3).
            points_2d (np.ndarray): Array of 2D points in the image plane (Nx2).
            method (int): PnP method to use (e.g., cv2.SOLVEPNP_ITERATIVE, cv2.SOLVEPNP_EPNP).

        Returns:
            success (bool): True if the pose estimation was successful.
            rvec (np.ndarray): Rotation vector (3x1).
            tvec (np.ndarray): Translation vector (3x1).
        """
        # Convert inputs to numpy arrays if they are not already
        points_3d = np.array(points_3d, dtype=np.float32)
        points_2d = np.array(points_2d, dtype=np.float32)

        # Solve PnP problem
        success, rvec, tvec = cv2.solvePnP(
            points_3d, points_2d, self.camera_matrix, self.dist_coeffs, flags=method
        )

        return success, rvec, tvec

    def project_points(self, points_3d, rvec, tvec):
        """
        Project 3D points to 2D image plane using the estimated camera pose.

        Args:
            points_3d (np.ndarray): Array of 3D points in the world coordinate system (Nx3).
            rvec (np.ndarray): Rotation vector (3x1).
            tvec (np.ndarray): Translation vector (3x1).

        Returns:
            points_2d (np.ndarray): Projected 2D points in the image plane (Nx2).
        """
        points_2d, _ = cv2.projectPoints(
            points_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs
        )
        return points_2d.reshape(-1, 2)

    def rotation_vector_to_matrix(self, rvec):
        """
        Convert a rotation vector to a rotation matrix.

        Args:
            rvec (np.ndarray): Rotation vector (3x1).

        Returns:
            rotation_matrix (np.ndarray): 3x3 rotation matrix.
        """
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        return rotation_matrix

    def get_camera_pose(self, rvec, tvec):
        """
        Get the camera pose as a 4x4 transformation matrix.

        Args:
            rvec (np.ndarray): Rotation vector (3x1).
            tvec (np.ndarray): Translation vector (3x1).

        Returns:
            pose_matrix (np.ndarray): 4x4 transformation matrix.
        """
        rotation_matrix = self.rotation_vector_to_matrix(rvec)
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = rotation_matrix
        pose_matrix[:3, 3] = tvec.flatten()
        return pose_matrix

if __name__ == "__main__":
    # Example camera intrinsic matrix (fx, fy, cx, cy)
    camera_matrix = np.array([
        [800, 0, 320],
        [0, 800, 240],
        [0, 0, 1]
    ])

    # Example distortion coefficients (k1, k2, p1, p2, k3)
    dist_coeffs = np.array([-0.1, 0.01, 0, 0, 0])

    # Pose estimation
    pose_estimator = CameraPoseEstimator(camera_matrix, dist_coeffs)

    # Example 3D points in the world coordinate system
    points_3d = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
    ], dtype=np.float32)

    # Example 2D points in the image plane
    points_2d = np.array([
        [320, 240],
        [400, 240],
        [320, 320],
        [280, 200],
        [400, 320],
        [400, 320]
    ], dtype=np.float32)

    # Estimate camera pose
    success, rvec, tvec = pose_estimator.estimate_pose(points_3d, points_2d)

    if success:
        print("Rotation Vector (rvec):\n", rvec)
        print("Translation Vector (tvec):\n", tvec)

        # Get camera pose as a 4x4 transformation matrix
        pose_matrix = pose_estimator.get_camera_pose(rvec, tvec)
        print("Camera Pose Matrix:\n", pose_matrix)

        # Project 3D points to 2D image plane
        projected_points = pose_estimator.project_points(points_3d, rvec, tvec)
        print("Projected 2D Points:\n", projected_points)