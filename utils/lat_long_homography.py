import cv2
import numpy as np
import math
from tqdm import tqdm

def projectPoint(point, h, offset_product=[1, 1], offset_sum=[0, 0]):
    p = np.array((point[0], point[1], 1)).reshape((3,1))
    temp_p = h.dot(p)
    sum = np.sum(temp_p ,1)
    px = sum[0]/sum[2] * offset_product[0] + offset_sum[0]
    py = sum[1]/sum[2] * offset_product[1] + offset_sum[1]
    return px, py

def getDistance(point_1, point_2):
    R = 6371e3
    theta_1 = point_1[0] * math.pi/180
    theta_2 = point_2[0] * math.pi/180
    delta_lat = (point_2[0] - point_1[0]) * math.pi/180
    delta_long = (point_2[1] - point_1[1]) * math.pi/180

    a = math.sin(delta_lat/2) * math.sin(delta_lat/2) + math.cos(theta_1) * math.cos(theta_2) * \
        math.sin(delta_long/2) * math.sin(delta_long/2)
    c = math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c
    return d

def getProjectionError_distance(src_points, dst_points, homo_mtx, offset_product=[1, 1], offset_sum=[0, 0]):
    src_points_withZ = np.hstack((src_points, np.ones((src_points.shape[0], 1), dtype=src_points.dtype))).reshape(-1, 3, 1)

    temp_p = homo_mtx.dot(src_points_withZ)
    sum = np.sum(temp_p ,2)
    px = (sum[0]/sum[2]) * offset_product[0] + offset_sum[0]
    py = sum[1]/sum[2] * offset_product[1] + offset_sum[1]
    
    transformed_points = np.concatenate((np.expand_dims(px, 1), np.expand_dims(py, 1)), 1)

    # Calculate distances between points    
    R = 6371e3
    theta_1 = transformed_points[:, 0] * np.pi/180
    theta_2 = dst_points[:, 0] * np.pi/180
    delta_lat = (dst_points[:, 0] - transformed_points[:, 0]) * np.pi/180
    delta_long = (dst_points[:, 1] - transformed_points[:, 1]) * np.pi/180

    a = np.sin(delta_lat/2) * np.sin(delta_lat/2) + np.cos(theta_1) * np.cos(theta_2) * \
        np.sin(delta_long/2) * np.sin(delta_long/2)

    c = np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = R * c

    return d
    
if __name__ == '__main__' :
    # camera points
    pts_src = np.array([
        [244,	174],
        [224,	230],
        [135,	374],
        [339,	433],
        [297,	495],
        [410,	800],
        [678,	740],
        [682,	871],
        [851,	692],
        [1022,	693],
        [1052,	898],
        [1364,	477],
        [1527,	474],
        [1549,	570],
        [1185,	530],
        [1207,	451],
        [1362,	404],
        [1567,	419],
        [1549,	336],
        [1252,	222],
        [1241,	294],
        [1074,	246],
        [1060,	296],
        [941,	236],
        [925,	271],
        [899,	328],
        [806,	514],
        [857,	414],
        [612,	508],
        [1149,	58],
        [1050,	47],
        [852,   53]
    ])

    # Map long, lat
    pts_dst = np.array([
        [48.7755537320585,  11.4251618004278],
        [48.77556477958027, 11.425100109620338],
        [48.775444582411495, 11.424962646408337],
        [48.775351782930436, 11.424975386901176],
        [48.775333664916495, 11.424929789347853],
        [48.77519534912667, 11.424816466017038],
        [48.775150274890656, 11.424890897317315],
        [48.77511845775853, 11.424838594241443],
        [48.77512155109042, 11.424941859288717],
        [48.77508973394008, 11.424987456842038],
        [48.77504377579844, 11.424891567869613],
        [48.77504996247188, 11.425163812084715],
        [48.77501019097409, 11.425194657488433],
        [48.774989421401635, 11.425134978337763],
        [48.77508089583725, 11.42509541575473],
        [48.775094594893, 11.425148389382857],
        [48.775065429156946, 11.42521276239931],
        [48.775009749068396, 11.425240255041755],
        [48.77503228625442, 11.425309321923864],
        [48.77514894917565, 11.425354919477035],
        [48.7751290634699, 11.425283840938034],
        [48.77519888435757, 11.425293899221854],
        [48.77518430152196, 11.425242937250493],
        [48.7752483775862, 11.42527646486323],
        [48.775237771898475, 11.42523824338471],
        [48.77522142145879, 11.425179234786295],
        [48.77518120819303, 11.425028360529145],
        [48.775200651974096, 11.425103462381678],
        [48.77524616806848, 11.425031713290421],
        [48.77526055916223, 11.425536836694087],
        [48.77530740091163, 11.425533483932815],
        [48.77539268813839, 11.425481180857041]
    ])
    # 48.77, 11.42
    pts_dst_decimal = np.array([
        [5.55373205850, 5.16180042780],
        [5.56477958027, 5.10010962033],
        [5.44458241150, 4.96264640833],
        [5.35178293044, 4.97538690117],
        [5.33366491650, 4.92978934785],
        [5.19534912667, 4.81646601703],
        [5.15027489066, 4.89089731731],
        [5.11845775853, 4.83859424144],
        # [48,775064, 11,424798],
        [5.12155109042, 4.94185928871],
        [5.08973394008, 4.98745684203],
        [5.04377579844, 4.89156786961],
        [5.04996247188, 5.16381208472],
        [5.01019097409, 5.19465748843],
        [4.98942140164, 5.13497833776],
        [5.08089583725, 5.09541575473],
        [5.09459489300, 5.14838938286],
        [5.06542915695, 5.21276239931],
        [5.00974906840, 5.24025504176],
        [5.03228625442, 5.30932192386],
        [5.14894917565, 5.35491947704],
        [5.12906346990, 5.28384093803],
        [5.19888435757, 5.29389922185],
        [5.18430152196, 5.24293725049],
        [5.24837758620, 5.27646486323],
        [5.23777189848, 5.23824338471],
        [5.22142145879, 5.17923478630],
        [5.18120819303, 5.02836052915],
        [5.20065197409, 5.10346238168],
        [5.24616806848, 5.03171329042],
        [5.26055916223, 5.53683669409],
        [5.30740091163, 5.53348393282],
        [5.39268813839, 5.48118085704]
    ])
    
    pts_dst_f32 = np.float32(pts_dst)
    pts_dst_decimal_f32 = np.float32(pts_dst_decimal)
    pts_src_f32 = np.float32(pts_src)
    
    TOTAL_NUM_SAMPLES = pts_src_f32.shape[0] # 32
    SAMPLE_SIZES = range(5, TOTAL_NUM_SAMPLES)

    total_lat_errors = []
    total_long_errors = []
    total_errors = []

    least_total_error = 1000
    final_data = None

    indices_list = np.arange(TOTAL_NUM_SAMPLES)

    for sample_size in tqdm(SAMPLE_SIZES):
        for iterator in tqdm(range(1000)):
            # print(f"Processing {sample_size}, {iterator}")
            selected_indices = np.random.choice(indices_list, sample_size)

            selected_src = np.take(pts_src_f32, selected_indices, axis=0)
            selected_dst = np.take(pts_dst_decimal_f32, selected_indices, axis=0)
            homo_mtx, status = cv2.findHomography(selected_src, selected_dst)

            total_error = np.percentile(getProjectionError_distance(pts_src_f32, pts_dst_f32, homo_mtx, offset_product=[1e-3, 1e-3], offset_sum=[48.77, 11.42]), 90)
            if total_error < least_total_error:
                least_total_error = total_error
                final_data = (least_total_error, np.copy(homo_mtx), sample_size)

    print(f"Average error (in m): {final_data[0]}")
    print(f"Homography mtx: {final_data[1]}")
    print(f"Number of samples used: {final_data[2]}")

    output_file_location = "map_files/homography_CameraToLatLong.npy" 
    print(f"Finished. Saved to {output_file_location}")
    np.save(output_file_location, final_data[1])