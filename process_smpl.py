import numpy as np
import pickle
from plyfile import PlyData

SMPL_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
]

PART_NAMES = [
    "background",
    "rightHand",
    "rightUpLeg",
    "leftArm",
    "head",
    "leftLeg",
    "leftFoot",
    "torso",
    "rightFoot",
    "rightArm",
    "leftHand",
    "rightLeg",
    "leftForeArm",
    "rightForeArm",
    "leftUpLeg",
    "hips",
]

JOINT_TO_PART = [
    15,
    14, 2,
    15,
    5, 11,
    7,
    6, 8,
    7,
    6, 8,
    4,
    7, 7,
    4,
    3, 9,  # shoulder
    12, 13,  # elbow'
    1, 10,  # wrist
    1, 10,  # hand
]

PART_COLOR = [
    [226, 226, 226],
    [158, 143, 143],  # rightHand
    [243, 115, 68],  # rightUpLeg
    [228, 162, 227],  # leftArm
    [210, 78, 142],  # head
    [152, 78, 163],  # leftLeg
    [76, 134, 26],  # leftFoot
    [100, 143, 255],  # torso
    [129, 0, 50],  # rightFoot
    [255, 176, 0],  # rightArm
    [192, 100, 119],  # leftHand
    [149, 192, 228],  # rightLeg
    [243, 232, 88],  # leftForeArm
    [90, 64, 210],  # rightForeArm
    [152, 200, 156],  # leftUpLeg
    [129, 103, 106],  # hips
]

# PART_COLOR_2 = [
#     [226, 226, 226],
#     [129, 0, 50],  # rightHand
#     [243, 115, 68],  # rightUpLeg
#     [228, 162, 227],  # leftArm
#     [210, 78, 142],  # head
#     [152, 78, 163],  # leftLeg
#     [76, 134, 26],  # leftFoot
#     [100, 143, 255],  # torso
#     [158, 143, 143],  # rightFoot
#     [255, 176, 0],  # rightArm
#     [192, 100, 119],  # leftHand
#     [149, 192, 228],  # rightLeg
#     [243, 232, 88],  # leftForeArm
#     [90, 64, 210],  # rightForeArm
#     [152, 200, 156],  # leftUpLeg
#     [129, 103, 106],  # hips
# ]


PART_COLOR_2 = [
    [0, 0, 0],  # 背景
    [0, 255, 255],  # rightHand
    [255, 165, 0],  # rightUpLeg
    [255, 192, 203],  # leftArm
    [255, 0, 0],  # head
    [128, 0, 128],  # leftLeg
    [169, 169, 169],  # leftFoot
    [144, 238, 144],  # torso
    [128, 128, 128],  # rightFoot
    [255, 140, 0],  # rightArm
    [255, 20, 147],  # leftHand
    [173, 216, 230],  # rightLeg
    [255, 255, 0],  # leftForeArm
    [0, 0, 255],  # rightForeArm
    [152, 251, 152],  # leftUpLeg
    [105, 105, 105],  # hips
]




def generate_ply(points, colors, labels):
    arr = [(points[i][0], points[i][1], points[i][2],
            colors[i][0], colors[i][1], colors[i][2],
            labels[i]) for i in range(len(points))]

    arr = np.array(arr, dtype=[
        ('x', 'f4'),
        ('y', 'f4'),
        ('z', 'f4'),
        ('red', 'uint8'),
        ('green', 'uint8'),
        ('blue', 'uint8'),
        ('label', 'uint8')])

    from plyfile import PlyElement
    el = PlyElement.describe(arr, 'vertex')
    plydata = PlyData([el], False, '<')

    return plydata


def save_plyfile(plydata: PlyData, file_path):
    print(f'save to {file_path}')
    with open(file_path, "wb") as f:
        plydata.text = False
        plydata.write(f)


def get_colors_from_labels(labels):
    colors = []
    for label in labels:
        colors.append(PART_COLOR_2[label])
    return colors


if __name__ == '__main__':
    from smpl.smpl_numpy import SMPL

    smpl_path = './assets/SMPL_NEUTRAL.pkl'
    save_path = './assets/smpl_semantic_big_pose.ply'
    with open(smpl_path, 'rb') as f:
        params = pickle.load(f, encoding="latin1")
        smpl_weights = params['weights']
        smpl_v_template = params['v_template']

        big_pose_smpl_param = {}
        big_pose_smpl_param['R'] = np.eye(3).astype(np.float32)
        big_pose_smpl_param['Th'] = np.zeros((1, 3)).astype(np.float32)
        big_pose_smpl_param['shapes'] = np.zeros((1, 10)).astype(np.float32)
        big_pose_smpl_param['poses'] = np.zeros((1, 72)).astype(np.float32)
        big_pose_smpl_param['poses'][0, 5] = 45 / 180 * np.array(np.pi)
        big_pose_smpl_param['poses'][0, 8] = -45 / 180 * np.array(np.pi)
        big_pose_smpl_param['poses'][0, 23] = -30 / 180 * np.array(np.pi)
        big_pose_smpl_param['poses'][0, 26] = 30 / 180 * np.array(np.pi)

        smpl_model = SMPL(sex='neutral', model_dir='assets/SMPL_NEUTRAL.pkl')

        big_pose_xyz, _ = smpl_model(big_pose_smpl_param['poses'], big_pose_smpl_param['shapes'].reshape(-1))
        big_pose_xyz = (
                    np.matmul(big_pose_xyz, big_pose_smpl_param['R'].transpose()) + big_pose_smpl_param['Th']).astype(
            np.float32)

    labels = []
    for w in smpl_weights:
        labels.append(JOINT_TO_PART[np.argmax(w)])

    colors = get_colors_from_labels(labels)

    ply = generate_ply(big_pose_xyz, colors, labels)
    # ply = generate_ply(smpl_v_template,colors,labels)
    save_plyfile(ply, save_path)