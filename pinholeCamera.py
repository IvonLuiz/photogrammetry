import numpy as np
import matplotlib.pyplot as plt
import cv2


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
        Projects a 3d array of points into camera plane considering lens distortions.
        
        Parameters:
        - world_pts_3d: Array of 3D points (Nx3).

        Returns:
        - Array of 2D projected points (Nx2).
        """
        camera_matrix = np.eye(4)
        camera_matrix[:3, :3] = self.world_R_cam
        camera_matrix[:3, 3] = self.world_T_cam.flatten()
        
        X = np.ones((world_pts_3d.shape[0], 4))
        X[:, :3] = world_pts_3d
        
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
        
        # Apply lens distortion
        k1, k2, p1, p2, k3 = self.dist_coefs
        x = img_pts_2d[:, 0]
        y = img_pts_2d[:, 1]
        r2 = x**2 + y**2
        r4 = r2**2
        r6 = r2**3

        # Radial distortion
        radial_distortion = 1 + k1 * r2 + k2 * r4 + k3 * r6

        # Tangential distortion
        x_tangential = 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
        y_tangential = p1 * (r2 + 2 * y**2) + 2 * p2 * x * y

        # Apply distortions
        x_distorted = x * radial_distortion + x_tangential
        y_distorted = y * radial_distortion + y_tangential

        img_pts_2d_distorted = np.vstack((x_distorted, y_distorted)).T

        return img_pts_2d_distorted

    
    def unproject(self, image_pts_2d: np.ndarray) -> np.ndarray:
        """
        Projects 2D detected points from the image as camera rays.

        Parameters:
        - image_pts_2d: Array of 2D points (Nx2).

        Returns:
        - Array of 3D direction vectors in the camera coordinate system (Nx3).
        """
        # Undistort points
        undistorted_pts = cv2.undistortPoints(image_pts_2d.reshape(-1, 1, 2), self.K, self.dist_coefs).reshape(-1, 2)

        # Normalize 2D points using intrinsic matrix (K)
        # Xc = K⁻¹ x Zc
        K_inv = np.linalg.inv(self.K)
        undistorted_pts_hom = np.hstack((undistorted_pts, np.ones((undistorted_pts.shape[0], 1))))
        cam_rays = (K_inv @ undistorted_pts_hom.T).T

        return cam_rays

        
# TESTING
def test_pinhole_camera():
    # Camera configuration
    K = np.array([[800, 0, 320],
                  [0, 800, 240],
                  [0, 0, 1]])
    
    no_dist_coefs = np.array([0, 0, 0, 0, 0])
    dist_coefs = np.array([0.00, -0.00, 0, 0.01, 0.0])  
   
    world_R_cam = np.eye(3)
    world_T_cam = np.array([[0], [0], [-10]])  # Translate the camera to -10 on the Z axis
    
    camera = PinholeCamera(K, no_dist_coefs, world_R_cam, world_T_cam)
    camera_with_dist = PinholeCamera(K, dist_coefs, world_R_cam, world_T_cam)
    
    # 3D Cube
    cube_vertices = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1]
    ])
    cube_edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]

    projected_pts = camera.project(cube_vertices)
    projected_pts_with_dist = camera_with_dist.project(cube_vertices)
    
    # Unproject
    unprojected_rays = camera.unproject(projected_pts)
    print("Unprojected Rays:")
    print(unprojected_rays)
    
    # Plot
    fig = plt.figure(figsize=(14, 7))
    ax3d = fig.add_subplot(131, projection='3d')
    
    for edge in cube_edges:
        ax3d.plot(
            cube_vertices[edge, 0], cube_vertices[edge, 1], cube_vertices[edge, 2], color='blue'
        )
    ax3d.set_title("Cube 3D")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")

    # Plot da projeção 2D sem distorção
    ax2d_no_dist = fig.add_subplot(132)
    for edge in cube_edges:
        start_idx, end_idx = edge
        start_point = projected_pts[start_idx]
        end_point = projected_pts[end_idx]
        ax2d_no_dist.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color='red')
    ax2d_no_dist.set_title("Projection 2D no Distorção")
    ax2d_no_dist.set_xlabel("X")
    ax2d_no_dist.set_ylabel("Y")
    ax2d_no_dist.set_xlim(0, 640)
    ax2d_no_dist.set_ylim(480, 0)  # Invertendo o eixo Y para coordenadas de imagem

    # Plot da projeção 2D com distorção
    ax2d_with_dist = fig.add_subplot(133)
    for edge in cube_edges:
        start_idx, end_idx = edge
        start_point = projected_pts_with_dist[start_idx]
        end_point = projected_pts_with_dist[end_idx]
        ax2d_with_dist.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color='green')
    ax2d_with_dist.set_title("Projection 2D with Distorção")
    ax2d_with_dist.set_xlabel("X")
    ax2d_with_dist.set_ylabel("Y")
    ax2d_with_dist.set_xlim(-5000, 6400)
    ax2d_with_dist.set_ylim(4800, -4000)  # Invertendo o eixo Y para coordenadas de imagem

    plt.tight_layout()
    plt.savefig("test_pinhole_camera_with_distortion.png")

test_pinhole_camera()