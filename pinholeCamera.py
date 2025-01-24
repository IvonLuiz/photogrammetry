import numpy as np


class PinholeCamera():
    def __init__(self, K : np.ndarray, dist_coefs : np.ndarray, world_R_cam : np.ndarray, world_T_cam : np.ndarray):
        """
        Constructor, that takes camera intrinsic and extrinsics
        
        - Intrinsic parameters are specific to a camera. They include information like focal length (fx, fy)
        and optical centers (cx, cy). 
        camera matrix = [ fx   0  fx ]
                        [  0  fy  cy ]
                        [  0   0   1 ]
        
        -Extrinsic parameters corresponds to rotation and translation vectors which translates
        a coordinates of a 3D point to a coordinate system.
        
        Parameters:
        - K: Camera intrinsic matrix (3x3).
        - dist_coefs: Distortion coefficients (1D array, typically [k1, k2, p1, p2, k3]).
        - world_R_cam: Camera rotation matrix (3x3).
        - world_T_cam: Camera translation vector (3x1).
        """
        
        self.K = K
        self.dist_coefs = dist_coefs
        self.world_R_cam = world_R_cam
        self.world_T_cam = world_T_cam
    
    def project(self, world_pts_3d : np.ndarray) -> np.ndarray:
        """
        Projects a 3d array of points into camera plane.
        
        Parameters:
        - world_pts_3d: Array of 3D points (Nx3).

        Returns:
        - Array of 2D projected points (Nx2).
        """
        camera_matrix = np.eye(4)
        camera_matrix[:3, :3] = self.world_R_cam
        camera_matrix[:3, 3] = self.world_T_cam.flatten()
        
        X = np.hstack((world_pts_3d, np.ones((world_pts_3d.shape[0], 1))))
        
        # Project into the image plane using the pinhole camera model:
        # x = K [R | T] X
        # Where:
        # - x: is the projected 2D point in homogeneous coordinates (image plane).
        # - K: is the intrinsic matrix (camera calibration matrix).
        # - R: is the rotation matrix (from world to camera coordinates).
        # - T: is the translation vector (from world to camera coordinates).
        # - X: is the 3D point in homogeneous coordinates (world coordinates).
        cam_coords = camera_matrix @ X.T  # (4xN)
        
        # Apply intrinsic matrix to get image coordinates
        img_coords_hom = self.K @ cam_coords[:3, :]  # (3xN)

        # Normalize by the third row to get pixel coordinates
        img_pts_2d = (img_coords_hom[:2, :] / img_coords_hom[2, :]).T  # (Nx2)
        print(img_coords_hom)
        print(img_pts_2d)
        return img_pts_2d
    

        
        
# TESTING
def test_pinhole_camera():
    # Intrinsic matrix
    K = np.array([[1000, 0, 320],
                  [0, 1000, 240],
                  [0, 0, 1]])

    # Distortion coefficients (not used in this test but included for completeness)
    dist_coefs = np.zeros(5)

    # Extrinsics (rotation and translation)
    world_R_cam = np.eye(3)  # Identity matrix (no rotation)
    world_T_cam = np.array([[0], [0], [-5]])  # Camera translated 5 units along Z-axis
    print("world T cam", world_R_cam)
    camera = PinholeCamera(K, dist_coefs, world_R_cam, world_T_cam)
    world_pts_3d = np.array([[0, 0, 0],
                             [1, 1, 0],
                             [-1, -1, 0],
                             [0.5, 0.5, 0],
                             [-0.5, -0.5, 0]])

    # Project the 3D points to the image plane
    projected_pts = camera.project(world_pts_3d)
    print("Projected Points:")
    print(projected_pts)


test_pinhole_camera()
