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