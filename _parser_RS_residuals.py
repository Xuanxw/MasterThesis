import os


def _parse_RS_residuals(file_path: str):
    pt_residuals_list = []
    with open(file_path, 'r') as res_file:
        res_lines = res_file.readlines()[1:32]
        for line in res_lines:
            l1 = line.rstrip().split()
            if l1 is not []:
                pt_residuals = [float(l) for l in l1[2:] if l]
                pt_residuals.append(int(l1[0]))
                pt_residuals_list.append(pt_residuals)
    return pt_residuals_list


if __name__ == '__main__':
    Residual_RS_4gcp = _parse_RS_residuals(
        'MasterThesis/data/UCE_M3_dataset/RS_projects/4GCP_Residuals.txt')
