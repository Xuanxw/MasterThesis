"""
A class to parse the project results from Inpho project's aat.log file
"""

import os
import numpy as np
from numpy import pi, sin, cos
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Inpho_prj_accuracy:
    def __init__(self, project_path: str, gcp_path: str):
        self.project_path = project_path
        self.pt_path = gcp_path
        self.pt_coordinates = []
        self.log_lines = []  # all the content in aat.log file
        self.tp_lines = []
        self.index_in_lines = []  # conduct the specific index of the lines in the log file
        self.img_id = []
        self.img_std = []  # standard deviation of all the images
        self.EO_residual = []
        self.tie_point_id = []
        self.tie_points_adjusted_coordinates = []
        self.cp_id = []  # id of the check points
        self.cp_residual = np.array([])  # residuals of the check points
        self.cp_RMS = np.array([])
        self.gcp_id = []  # id of the ground control points
        # residuals of the ground control points
        self.gcp_residual = np.array([])
        self.gcp_RMS = np.array([])
        self.imgstd_breaklines = 4
        self.ptres_breaklines = 6
        self.eores_imu_breaklines = 6
        self.eores_gnss_breaklines = 11

    def find_ind(self, arg: str, lines: list, breaklines=1, end_args='') -> list:
        index_in_lines = []
        ind_start = 0
        ind_end = 0
        for i, line1 in enumerate(lines):
            if arg in line1.rstrip():
                ind_start = i + breaklines
                for j, line2 in enumerate(lines):
                    if j > ind_start and line2.rstrip() == end_args:
                        ind_end = j
                        break
                index_in_lines.extend(np.arange(ind_start, ind_end))
            else:
                continue
        self.index_in_lines = index_in_lines
        return self.index_in_lines

    def parse_project_log(self) -> list:
        log_path = os.path.join(self.project_path, 'aat.log')
        with open(log_path) as log_file:
            self.log_lines = log_file.readlines()
        return None

    def parse_gcp_coordinates(self):
        with open(self.pt_path) as gcp_file:
            pt_lines = gcp_file.readlines()[1:]
            for line in pt_lines:
                l1 = line.rstrip().split()
                pt_coordinate = [float(l) for l in l1[1:] if l]
                pt_coordinate.append(int(l1[0]))
                self.pt_coordinates.append(pt_coordinate)
        return None

    # optional
    def parse_current_tie_points_from_xpf_file(self):
        for file in os.listdir(self.project_folder):
            if file.endswith(".xpf"):
                xpf_path = os.path.join(self.project_folder, file)

        if os.path.isfile(xpf_path):
            with open(xpf_path) as tp_file:
                self.tp_lines = tp_file.readlines()
            index_tp_line = self.find_ind(arg='$ADJUSTED_POINTS',
                                          lines=self.tp_lines, breaklines=1, end_args='$END_POINTS')
            for i in index_tp_line:
                tie_point = list(
                    filter(None, self.tp_lines[i].strip().split(' ')))
                self.tie_point_id.append(int(tie_point[0]))
                self.tie_points_adjusted_coordinates.append(
                    [float(tie_point[1]), float(tie_point[2]), float(tie_point[3])])
        return None

    def parse_std_img(self):
        ind_std = self.find_ind(
            'standard deviations of exterior orientation parameters (px, py, pz in [meter] omega,phi,kappa in [grd/1000] )',
            self.log_lines,
            self.imgstd_breaklines)
        img_std = []
        for i in ind_std:
            img_std = list(filter(None, self.log_lines[i].strip().split(' ')))
            self.img_std.append([float(x) for x in img_std if x != img_std[0]])
            self.img_id.append(img_std[0])
        return None

    def parse_res_EO(self):
        ind_res_pos = self.find_ind(
            'residuals  GNSS observations in [meter]', self.log_lines, self.eores_gnss_breaklines)
        ind_res_rot = self.find_ind(
            'residuals  IMU observations in [grd]', self.log_lines, self.eores_imu_breaklines)
        if ind_res_rot:
            for i, j in zip(ind_res_pos, ind_res_rot):
                img_res_pos = list(
                    filter(None, self.log_lines[i].strip().split(' ')))
                img_res_rot = list(
                    filter(None, self.log_lines[j].strip().split(' ')))
                self.EO_residual.append([float(x) for x in img_res_pos if x != img_res_pos[0]] +
                                        [float(y)for y in img_res_rot if y != img_res_rot[0]])
        else:
            for i in ind_res_pos:
                img_res_pos = list(
                    filter(None, self.log_lines[i].strip().split(' ')))
                self.EO_residual.append([float(x) for x in img_res_pos if x != img_res_pos[0]] +
                                        [0.0, 0.0, 0.0])
        return None

    def parse_RMS_pt(self):
        ind_RMS_gcp = self.find_ind('RMS control points', self.log_lines)
        ind_RMS_cp = self.find_ind('RMS at check  points', self.log_lines)
        RMS_gcp_alaxis = []
        RMS_cp_alaxis = []
        for i in ind_RMS_gcp:
            RMS_gcp = float(self.log_lines[i].strip().split()[1])
            RMS_gcp_alaxis.append(RMS_gcp)
        self.gcp_RMS = np.array(RMS_gcp_alaxis)
        for j in ind_RMS_cp:
            RMS_cp = float(self.log_lines[j].strip().split()[1])
            RMS_cp_alaxis.append(RMS_cp)
        self.cp_RMS = np.array(RMS_cp_alaxis)
        return None

    def parse_res_pt(self):
        ind_resXY = self.find_ind(
            'residuals  horizontal control points in [meter]',
            self.log_lines, self.ptres_breaklines)
        ind_resZ = self.find_ind(
            'residuals  vertical control points in [meter]',
            self.log_lines,
            self.ptres_breaklines)
        cp_residual = []
        gcp_residual = []
        for i, j in zip(ind_resXY, ind_resZ):
            # Check point
            if 'check point' in self.log_lines[i]:
                cp_resXY = list(
                    filter(None, self.log_lines[i].strip().split(' ')))
                cp_resZ = list(
                    filter(None, self.log_lines[j].strip().split(' ')))
                self.cp_id.append(int(cp_resXY[0]))
                if self.pt_coordinates:
                    for pt_pos in self.pt_coordinates:
                        if pt_pos[-1] is int(cp_resXY[0]):
                            cp_residual.append([float(cp_resXY[1]), float(cp_resXY[2]),
                                               float(cp_resZ[1]), pt_pos[0], pt_pos[1], pt_pos[2]])
                self.cp_residual = np.array(cp_residual)
                continue
            # Ground control point
            else:
                gcp_resXY = list(
                    filter(None, self.log_lines[i].strip().split(' ')))
                gcp_resZ = list(
                    filter(None, self.log_lines[j].strip().split(' ')))
                self.gcp_id.append(int(gcp_resXY[0]))
                if self.pt_coordinates:
                    for pt_pos in self.pt_coordinates:
                        if pt_pos[-1] is int(gcp_resXY[0]):
                            gcp_residual.append([float(gcp_resXY[1]), float(gcp_resXY[2]),
                                                 float(gcp_resZ[1]), pt_pos[0], pt_pos[1], pt_pos[2]])
                self.gcp_residual = np.array(gcp_residual)
                continue
        return None

    def run(self):
        _ = self.parse_project_log()
        _ = self.parse_gcp_coordinates()
        _ = self.parse_RMS_pt()
        _ = self.parse_std_img()
        _ = self.parse_res_pt()
        _ = self.parse_res_EO()
        # _ = self.parse_current_tie_points_from_ascii_file()
        return None


class Exterior_Orientation_Parser:
    def __init__(self, ori_file_path: str, navi_file_path: str):
        self.ori_path = ori_file_path
        self.navi_path = navi_file_path
        self.image_eo_navi = []
        self.image_eo_post = []

    def parse_calculated_EO_from_ori_file(self) -> None:
        with open(self.ori_path) as ori_file:
            ori_lines = ori_file.readlines()
        for line in ori_lines:
            if not line.rstrip().startswith('#'):
                line_strip = line.rstrip().split()
                eo_post = [float(x) for x in line_strip if x != line_strip[0]]
                eo_post[3:] = deg2grad(eo_post[3:])
                self.image_eo_post.append(eo_post)
        return None

    def parse_original_EO_from_navi_file(self) -> None:
        with open(self.navi_path) as navi_file:
            navi_lines = navi_file.readlines()
        for line2 in navi_lines:
            if not line2.rstrip().startswith('#') and line2.rstrip() != '':
                eo_navi = line2.rstrip().split()
                eo_navi[2] = eo_navi[2][2:]
                self.image_eo_navi.append(
                    [float(x) for x in eo_navi if x != eo_navi[0] and x != eo_navi[1]])
        return None

    def run(self):
        _ = self.parse_calculated_EO_from_ori_file()
        _ = self.parse_original_EO_from_navi_file()


def ellipse(center_x: float, center_y: float, a: float, b: float, ax1=[1, 0], ax2=[0, 1], N=500):
    t = np.linspace(0, 2 * pi, N)
    xs = a * cos(t)
    ys = b * sin(t)
    R = np.array([ax1, ax2]).T
    xp, yp = np.dot(R, [xs, ys])
    x = xp + center_x
    y = yp + center_y
    return [x, y]


def deg2grad(deg_value_list: list) -> list:
    grad_value_list = [deg_value * 200.0 /
                       180.0 for deg_value in deg_value_list]
    return grad_value_list


if __name__ == '__main__':
    GCP_path = 'MasterThesis/data/UCE_M3_dataset/PP_CP_Koordinaten_1.txt'
    project_path = 'MasterThesis/data/UCE_M3_dataset/Inpho_projects/project_UCE_M3_rgb-15GCP+18CP'
    prj = Inpho_prj_accuracy(
        project_path=project_path, gcp_path=GCP_path)
    _ = prj.run()
    fig = go.Figure()
