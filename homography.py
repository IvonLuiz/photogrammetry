import numpy as np
import cv2

class CameraPoseEstimatorHomography:
    def __init__(self, marker_size, camera_matrix, dist_coeffs=None):
        """
        Initialize the CameraPoseEstimator.

        Args:
            marker_size (tuple): Size of the planar marker in meters (width, height).
            camera_matrix (np.ndarray): 3x3 camera intrinsic matrix.
            dist_coeffs (np.ndarray, optional): Distortion coefficients. Defaults to None.
        """
        self.marker_size = marker_size
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros((4, 1))

        # Define 3D points of the planar marker in the world coordinate system
        self.marker_points_3d = np.array([
            [0, 0, 0],  # Bottom-left corner
            [marker_size[0], 0, 0],  # Bottom-right corner
            [marker_size[0], marker_size[1], 0],  # Top-right corner
            [0, marker_size[1], 0]  # Top-left corner
        ], dtype=np.float32)

    def estimate_pose(self, marker_corners_2d):
        """
        Estimate the camera pose from the 2D corners of the planar marker.

        Args:
            marker_corners_2d (np.ndarray): 2D corners of the marker in the image (4x2).

        Returns:
            success (bool): True if the pose estimation was successful.
            rvec (np.ndarray): Rotation vector (3x1).
            tvec (np.ndarray): Translation vector (3x1).
        """
        
        # COmpute the homography matrix
        H, _ = cv2.findHomography(self.marker_points_3d[:, :2], marker_corners_2d)
        
        # Decompose the hoography to get the rotation and translation
        num_solutions, rotations, translations, normals = cv2.decomposeHomographyMat(H, self.camera_matrix)

        # Select the correct solution (usually the first one)
        rvec, _ = cv2.Rodrigues(rotations[0])
        tvec = translations[0]
        
        return True, rvec, tvec

    def project_points(self, points_3d, rvec, tvec):
        """
        Project 3D points to the 2D image plane using the estimated camera pose.

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

    
    
if __name__ == "__main__":
    # Define the camera intrinsic matrix
    camera_matrix = np.array([
        [1000, 0, 640],
        [0, 1000, 360],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    # Physical size of the marker (e.g., 0.1 meters x 0.1 meters)
    marker_size = (0.1, 0.1)

    pose_estimator = CameraPoseEstimatorHomography(marker_size, camera_matrix, dist_coeffs)

    # Example marker corners in the image
    marker_corners = np.array([
        [100, 150],  # Bottom-left corner
        [300, 150],  # Bottom-right corner
        [300, 350],  # Top-right corner
        [100, 350]  # Top-left corner
    ], dtype=np.float32)


    success, rvec, tvec = pose_estimator.estimate_pose(marker_corners)

    if success:
        print("Rotation Vector (rvec):\n", rvec)
        print("Translation Vector (tvec):\n", tvec)

        # Example: project a 3D point in the marker's coordinate system onto the image
        obj_points = np.array([[0.1, 0.1, 0]], dtype=np.float32)  # A point on the marker
        img_points = pose_estimator.project_points(obj_points, rvec, tvec)
        print("Projected Point in Image:", img_points)
    
    else:
        print("Pose estimation failed.")
