data_path = "MasterThesis/data/EO_230603_AdVTestfeld_Neumünster_RGBI.txt"
gnss_observation = []
with open(data_path, 'r', encoding='utf-8', errors='ignore') as data_file:
    data_lines = data_file.readlines()

for line in data_lines:
    if not line.rstrip().startswith('#') and line.rstrip() != '':
        observation = line.rstrip().split()
        observation[1] = observation[1][2:]
        if len(observation[0]) < 3:
            observation[0] = '0' + observation[0]
        gnss_observation.append(observation)
# print(gnss_observation)
with open("MasterThesis/data/EO_230603_AdVTestfeld_Neumünster_RGBI_without32zone.txt", "w+")as new_file:
    for line in gnss_observation:
        new_file.write(" ".join(line))
        new_file.write("\n")
    new_file.close()

# Modify Point Coordinates
# data_path = "MasterThesis/data/UCE_M3_dataset/CP_Koordinaten.txt"
# points = []
# i = 0
# with open(data_path, 'r', encoding='utf-8', errors='ignore') as data_file:
#     data_lines = data_file.readlines()

# for line in data_lines:
#     if i != 0:
#         point = line.rstrip().split()
#         point[1] = point[1][2:]
#         for i, string in enumerate(point):
#             point[i] = point[i].replace(",", ".")
#         points.append(point)
#     i = +1

# with open("MasterThesis/data/UCE_M3_dataset/CP_Koordinaten_1.txt", "w+")as new_file:
#     for line in points:
#         new_file.write(" ".join(line))
#         new_file.write("\n")
#     new_file.close()
