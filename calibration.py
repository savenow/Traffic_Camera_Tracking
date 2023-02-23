import numpy as np

class Calibration():
    def __init__(self):
        self.OFFSET_X_WORLD = 678000
        self.OFFSET_Y_WORLD = 5400000

        self.camera_mtx = np.array([
            [1.95545129e+03, 0.00000000e+00, 9.60292435e+02],
            [0.00000000e+00, 2.22564136e+03, 6.27572239e+02],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ])
        self.rotation_mtx = np.array([
            [ 0.59323544, -0.80111521, -0.07928514],
            [-0.48164592, -0.27428963, -0.83233551],
            [ 0.64504955,  0.53195829, -0.5485722 ]
        ])
        self.translation_mtx = np.array([
            [ 4144.77583445],
            [ 1528.92256713],
            [-2864.02553347]
        ])

    
    def projection_pixel_to_world(self, pixel_point):
        pixel_point = np.array([pixel_point[0], pixel_point[1], 1]).reshape(3, 1)

        position_inv_rotation_cam_pixel = np.linalg.inv(self.rotation_mtx).dot(
            np.linalg.inv(self.camera_mtx).dot(pixel_point))

        inv_r_and_translation = 0 + self.rotation_mtx.transpose().dot(self.translation_mtx)[2, 0]

        s = inv_r_and_translation / position_inv_rotation_cam_pixel[2, 0]

        position = s * position_inv_rotation_cam_pixel
        full_position = position - self.rotation_mtx.transpose().dot(self.translation_mtx)

        return ([full_position[0, 0] + self.OFFSET_X_WORLD, full_position[1, 0] + self.OFFSET_Y_WORLD, full_position[2, 0]])

    def projection_world_to_pixel(self, world_position):
        world_position.shape = (3, 1)
        world_position[0] = world_position[0] - self.OFFSET_X_WORLD
        world_position[1] = world_position[1] - self.OFFSET_Y_WORLD
        pixels_s = self.camera_mtx.dot(self.rotation_mtx.dot(world_position) + self.translation_mtx)
        pixel = pixels_s / pixels_s[2]
        pixel = np.around(pixel, 2).astype(dtype=np.float32)
        return np.asarray([pixel[0][0], pixel[1][0]], dtype=np.float32)

class Calibration_LatLong():
    """Calculated by manually selecting points between for the masked video
    """
    def __init__(self):
        self.homo_mtx = np.load("map_files/homography_CameraToLatLong.npy")
        self.offset_product=[1e-3, 1e-3]
        self.offset_sum=[48.77, 11.42]

    def getLatLong(self, img_point):
        point = np.array((img_point[0], img_point[1], 1)).reshape((3, 1))
        projection = self.homo_mtx.dot(point)
        sum = np.sum(projection, 1)
        px = (sum[0] / sum[2]) * self.offset_product[0] + self.offset_sum[0]
        py = (sum[1] / sum[2]) * self.offset_product[1] + self.offset_sum[1]
        return px, py

    def getDistance(self, img_point_1, img_point_2):
        point_1_lat, point_1_long = self.getLatLong(img_point_1)
        point_2_lat, point_2_long = self.getLatLong(img_point_2)
        
        # Calculate distances between points    
        R = 6371e3
        theta_1 = point_1_lat * np.pi/180
        theta_2 = point_2_lat * np.pi/180
        delta_lat = (point_2_lat - point_1_lat) * np.pi/180
        delta_long = (point_2_long - point_1_long) * np.pi/180

        a = np.sin(delta_lat/2) * np.sin(delta_lat/2) + np.cos(theta_1) * np.cos(theta_2) * \
            np.sin(delta_long/2) * np.sin(delta_long/2)
        
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        d = R * c

        return d