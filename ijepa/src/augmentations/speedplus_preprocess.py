import os
import json
import csv
import numpy as np
import cv2
import scipy.io
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

class SpeedDatasetGenerator:
    def __init__(self, root_dir='.', dataname='speedplus', domain='synthetic', 
                 json_file='train.json', camera_file='camera.json', 
                 keypoints_file='tangoPoints.mat', output_file='train.csv'):
        self.root_dir = root_dir
        self.dataname = dataname
        self.domain = domain
        self.json_file = json_file
        self.camera_file = camera_file
        self.keypoints_file = keypoints_file
        self.output_file = output_file
        
        self.json_path = os.path.join(self.root_dir, self.dataname, self.domain, self.json_file)
        self.camera_path = os.path.join(self.root_dir, self.dataname, self.camera_file)
        self.output_path = os.path.join(self.root_dir, self.dataname, self.domain, self.output_file)
        
        if os.path.isabs(self.keypoints_file):
            self.mat_path = self.keypoints_file
        else:
            self.mat_path = os.path.join(self.root_dir, self.keypoints_file)

    def load_camera_intrinsics(self):
        with open(self.camera_path, 'r') as f:
            data = json.load(f)
        
        K = np.array(data['cameraMatrix'], dtype=np.float32)
        dist = np.array(data['distCoeffs'], dtype=np.float32)
        return K, dist

    def load_tango_3d_keypoints(self):
        try:
            mat = scipy.io.loadmat(self.mat_path)
            possible_keys = ['keypoints', 'p_3D', 'tangoPoints', 'points', 'vertices']
            points = None
            for k in possible_keys:
                if k in mat:
                    points = mat[k]
                    break
            
            if points is None:
                keys = [k for k in mat.keys() if not k.startswith('__')]
                if keys:
                    points = mat[keys[0]]
                
            if points is None:
                 raise ValueError("Could not find keypoints variable in .mat file")

            points = np.array(points, dtype=np.float32)
            
            if points.shape[0] == 3 and points.shape[1] > 3:
                points = points.T
                
            return points
        except Exception as e:
            raise FileNotFoundError(f"Error loading {self.mat_path}: {e}")

    def project_keypoints(self, q_vbs2tango, r_Vo2To_vbs, K, dist, points_3d):
        q_scalar_last = np.array([q_vbs2tango[1], q_vbs2tango[2], q_vbs2tango[3], q_vbs2tango[0]])
        r_mat = R.from_quat(q_scalar_last).as_matrix()
        r_vec, _ = cv2.Rodrigues(r_mat)
        img_points, _ = cv2.projectPoints(points_3d, r_vec, r_Vo2To_vbs, K, dist)
        
        return img_points.reshape(-1, 2)

    def generate(self):
        print(f"Reading labels from: {self.json_path}")
        print(f"Reading intrinsics from: {self.camera_path}")
        print(f"Reading 3D keypoints from: {self.mat_path}")

        with open(self.json_path, 'r') as f:
            labels = json.load(f)
        
        K, dist = self.load_camera_intrinsics()
        points_3d = self.load_tango_3d_keypoints()

        print(f"Writing to: {self.output_path}")
        
        header = ['filename', 'cx', 'cy', 'w', 'h'] 
        header.extend(['q_w', 'q_x', 'q_y', 'q_z'])
        header.extend(['t_x', 't_y', 't_z'])

        with open(self.output_path, 'w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(header)

            for item in tqdm(labels, desc="Processing Images"):
                filename = os.path.join(self.domain, 'images', item['filename'])
                
                q_vec = np.array(item['q_vbs2tango_true'], dtype=np.float32)
                t_vec = np.array(item['r_Vo2To_vbs_true'], dtype=np.float32)

                kpts_2d = self.project_keypoints(q_vec, t_vec, K, dist, points_3d)

                x_min_tight = np.min(kpts_2d[:, 0])
                y_min_tight = np.min(kpts_2d[:, 1])
                x_max_tight = np.max(kpts_2d[:, 0])
                y_max_tight = np.max(kpts_2d[:, 1])

                width_tight = x_max_tight - x_min_tight
                height_tight = y_max_tight - y_min_tight

                margin_x = 0.10 * width_tight
                margin_y = 0.10 * height_tight

                xmin = x_min_tight - margin_x
                ymin = y_min_tight - margin_y
                xmax = x_max_tight + margin_x
                ymax = y_max_tight + margin_y

                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(1920, xmax)
                ymax = min(1200, ymax)

                final_w = xmax - xmin
                final_h = ymax - ymin
                cx = xmin + (final_w / 2.0)
                cy = ymin + (final_h / 2.0)

                row = [filename, cx, cy, final_w, final_h]
                row.extend(q_vec.tolist())
                row.extend(t_vec.tolist())

                csv_writer.writerow(row)

        print("Done.")

if __name__ == '__main__':
    generator = SpeedDatasetGenerator(
        root_dir='/media/mlr/New Volume/ARNABI_ARSH/speedplus/',
        dataname='speedplusv2',
        domain='synthetic',
        json_file='train.json',
        camera_file='camera.json',
        keypoints_file='speedplusv2/tangoPoints.mat',
        output_file='train_synthetic.csv'
    )
    
    generator.generate()