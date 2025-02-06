import numpy as np


class Triangulation:
    def __init__(self, camera_params1, camera_params2):
        """
        Initialize the class with camera parameters for two cameras.
        
        :param camera_params1: Dictionary containing intrinsic and extrinsic parameters for camera 1.
        :param camera_params2: Dictionary containing intrinsic and extrinsic parameters for camera 2.
        """
        self.camera_params1 = camera_params1
        self.camera_params2 = camera_params2
        
        # Compute the projection matrices for both cameras
        self.P1 = self._compute_projection_matrix(camera_params1)
        self.P2 = self._compute_projection_matrix(camera_params2)
    
    def _compute_projection_matrix(self, camera_params):
        """
        Compute the projection matrix from camera parameters.
        
        :param camera_params: Dictionary containing intrinsic and extrinsic parameters.
        :return: 3x4 projection matrix.
        """
        # Intrinsic matrix (K)
        K = np.array(camera_params['intrinsic'])
        
        # Extrinsic parameters (rotation matrix R and translation vector t)
        R = np.array(camera_params['rotation'])
        t = np.array(camera_params['translation'])
        
        # Compute the extrinsic matrix [R | t]
        extrinsic = np.hstack((R, t.reshape(-1, 1)))
        
        # Compute the projection matrix P = K * [R | t]
        P = K @ extrinsic
        
        return P
    
    def triangulate_points(self, points1, points2):
        """
        Triangulate 3D points from 2D correspondences in two images.
        
        :param points1: 2D points in the first image (Nx2 array).
        :param points2: 2D points in the second image (Nx2 array).
        :return: 3D points in world coordinates (Nx3 array).
        """
        points1_h = np.hstack((points1, np.ones((points1.shape[0], 1))))
        points2_h = np.hstack((points2, np.ones((points2.shape[0], 1))))
        
        points_3d = np.zeros((points1.shape[0], 3))
        
        # Triangulate each pair of points
        for i in range(points1.shape[0]):
            x1 = points1_h[i, 0]
            y1 = points1_h[i, 1]
            x2 = points2_h[i, 0]
            y2 = points2_h[i, 1]

            A = np.vstack((
                x1 * self.P1[2] - self.P1[0],
                y1 * self.P1[2] - self.P1[1],
                x2 * self.P2[2] - self.P2[0],
                y2 * self.P2[2] - self.P2[1]
            ))
        
            # The system AX = 0 is solved using Singular Value Decomposition (SVD).
            # The solution is the singular vector corresponding to the smallest singular value,
            # which represents the estimate of the 3D point homogeneous coordenates.
            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1]  # Solution is the last row of Vt
            
            points_3d[i, :] = X[:3] / X[3]
        
        return points_3d


if __name__ == "__main__":
    # Define camera parameters (intrinsic and extrinsic)
    camera_params1 = {
        'intrinsic': np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]]),  # K matrix
        'rotation': np.eye(3),
        'translation': np.array([0, 0, 0])
    }
    
    camera_params2 = {
        'intrinsic': np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]]),  # K matrix
        'rotation': np.eye(3),
        'translation': np.array([1, 0, 1])
    }
    
    # Testing with 2D points in both images (Nx2 arrays)
    points1 = np.array([[320, 240], [330, 250], [340, 260]])
    points2 = np.array([[310, 240], [310, 270], [330, 120]])
    
    triangulator = Triangulation(camera_params1, camera_params2)
    points_3d = triangulator.triangulate_points(points1, points2)
    
    print("3D Points:\n", np.round(points_3d, 2))