# -*- coding: utf-8 -*-

"""
S3DIS Test Result Visualization

Only visualize:
1. Original Scene (raw RGB)
2. Ground Truth
3. Prediction

Coordinate source:
    Original S3DIS dataset coord.npy

Label source:
    GT  -> original segment.npy
    Pred -> test result pred.npy
"""


import os
import numpy as np
import open3d as o3d
from datetime import datetime


# ============================================================
# 1. 配置
# ============================================================


area_name = "Area_1"
room_name = "office_9"
scene_name = f"{area_name}-{room_name}"

# ---------- 原始S3DIS数据 ----------
dataset_root = (
    "/media/cvmaster/1C0329EB7F42121A/"
    "xlh/Data/s3dis"
)
scene_root = os.path.join(
    dataset_root,
    area_name,
    room_name
)

# ---------- test输出 ----------
test_root = (
    "/media/cvmaster/1C0329EB7F42121A/"
    "xlh/gjl/Pointcept/"
    "exp/s3dis/s3dis-area1-best-test"
)


pred_root = os.path.join(
    test_root,
    "result"
)


pred_path = os.path.join(
    pred_root,
    f"{scene_name}_pred.npy"
)


# ============================================================
# 2. S3DIS类别
# ============================================================


num_classes = 20


class_names = [
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refridgerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture",
]


# 颜色固定
colors = np.array([

    [0.6, 0.8, 0.6],
    [0.6, 0.7, 0.9],
    [0.2, 0.4, 0.7],
    [0.9, 0.7, 0.5],
    [0.7, 0.7, 0.4],
    [0.5, 0.3, 0.2],
    [0.9, 0.6, 0.6],
    [0.7, 0.3, 0.2],
    [0.8, 0.7, 0.9],
    [0.6, 0.4, 0.7],
    [0.5, 0.8, 0.8],
    [0.4, 0.6, 0.9],
    [0.9, 0.7, 0.8],
    [0.9, 0.9, 0.6],
    [0.9, 0.6, 0.3],
    [0.8, 0.6, 0.8],
    [0.7, 0.8, 0.8],
    [0.3, 0.6, 0.3],
    [0.5, 0.5, 0.6],
    [0.4, 0.4, 0.6],

],dtype=np.float64)



# ============================================================
# 3. 读取原始数据
# ============================================================


coord_path = os.path.join(
    scene_root,
    "coord.npy"
)


color_path = os.path.join(
    scene_root,
    "color.npy"
)


segment_path = os.path.join(
    scene_root,
    "segment.npy"
)



print("="*80)
print("Loading raw S3DIS data")
print(coord_path)
print(color_path)
print(segment_path)
print("="*80)



coord = np.load(
    coord_path
)


rgb = np.load(
    color_path
)


gt = np.load(
    segment_path
).reshape(-1)



print("coord :",coord.shape)
print("color :",rgb.shape)
print("GT    :",gt.shape)



# ============================================================
# 4. 读取预测结果
# ============================================================


pred = np.load(
    pred_path
).reshape(-1)



print("Pred  :",pred.shape)



assert len(coord)==len(gt)
assert len(coord)==len(pred)



# ============================================================
# 5. Open3D工具函数
# ============================================================


def create_pcd(
        xyz,
        rgb
):

    pcd = o3d.geometry.PointCloud()

    pcd.points = (
        o3d.utility.Vector3dVector(
            xyz
        )
    )


    rgb = np.asarray(rgb)

    if rgb.max()>1:
        rgb = rgb/255.0


    pcd.colors = (
        o3d.utility.Vector3dVector(
            rgb
        )
    )


    return pcd



def create_label_pcd(
        xyz,
        labels
):


    point_color = np.zeros(
        (len(labels),3)
    )


    for i in range(num_classes):

        point_color[
            labels==i
        ] = colors[i]


    pcd = o3d.geometry.PointCloud()


    pcd.points = (
        o3d.utility.Vector3dVector(
            xyz
        )
    )


    pcd.colors = (
        o3d.utility.Vector3dVector(
            point_color
        )
    )


    return pcd



# def save_and_show(
#         pcd,
#         name
# ):
#
#     vis = o3d.visualization.Visualizer()
#
#     vis.create_window(
#         window_name=name,
#         width=1280,
#         height=960
#     )
#
#     vis.add_geometry(pcd)
#
#
#     opt = vis.get_render_option()
#
#     opt.point_size = 5
#
#
#     vis.poll_events()
#
#     vis.update_renderer()
#
#
#     save_path=os.path.join(
#         output_dir,
#         name+".png"
#     )
#
#
#     vis.capture_screen_image(
#         save_path,
#         do_render=True
#     )
#
#
#     ply_path=os.path.join(
#         output_dir,
#         name+".ply"
#     )
#
#
#     o3d.io.write_point_cloud(
#         ply_path,
#         pcd
#     )
#
#
#     print(
#         "saved:",
#         save_path
#     )
#
#
#     while True:
#
#         if not vis.poll_events():
#             break
#
#         vis.update_renderer()
#
#
#     vis.destroy_window()



# ============================================================
# 6. 输出目录
# ============================================================
# def create_visualizer(
#         pcd,
#         name
# ):
#
#     vis = o3d.visualization.Visualizer()
#
#     vis.create_window(
#         window_name=name,
#         width=1280,
#         height=960
#     )
#
#     vis.add_geometry(pcd)
#
#     opt = vis.get_render_option()
#     opt.point_size = 3
#
#     return vis
def create_visualizer(
        pcd,
        name
):

    vis = o3d.visualization.Visualizer()

    vis.create_window(
        window_name=name,
        width=1280,
        height=960
    )

    vis.add_geometry(pcd)

    opt = vis.get_render_option()

    if opt is not None:
        opt.point_size = 3


    return vis
def save_image(
        vis,
        name
):

    path=os.path.join(
        output_dir,
        name+".png"
    )

    vis.poll_events()
    vis.update_renderer()

    vis.capture_screen_image(
        path
    )

    print(
        "saved:",
        path
    )

output_dir=os.path.join(
    "./vis_result",
    scene_name+"_"+datetime.now().strftime("%Y%m%d_%H%M%S")
)


os.makedirs(
    output_dir,
    exist_ok=True
)



# ============================================================
# 7. 三个可视化
# ============================================================


# -------- 原始场景 --------

pcd_original=create_pcd(
    coord,
    rgb
)


# ============================================================
# 创建三个窗口
# ============================================================


vis_list=[]



# Original

vis_original=create_visualizer(
    pcd_original,
    "00 Original Scene"
)

vis_list.append(
    vis_original
)

# -------- GT --------


pcd_gt=create_label_pcd(
    coord,
    gt
)


# -------- Prediction --------


pcd_pred=create_label_pcd(
    coord,
    pred
)

# GT

vis_gt=create_visualizer(
    pcd_gt,
    "01 Ground Truth"
)

vis_list.append(
    vis_gt
)



# Prediction

vis_pred=create_visualizer(
    pcd_pred,
    "02 Prediction"
)

vis_list.append(
    vis_pred
)



# 保存图片

save_image(
    vis_original,
    "00_original_scene"
)


save_image(
    vis_gt,
    "01_ground_truth"
)


save_image(
    vis_pred,
    "02_prediction"
)



print(
    "三个窗口同时打开"
)



# ============================================================
# 统一事件循环
# ============================================================

# ============================================================
# Prediction窗口控制其他窗口同步
# ============================================================


# Prediction作为主控制窗口
master_vis = vis_pred


# 保存三个窗口初始状态
last_camera = None



while True:


    alive = False


    # ===============================
    # 处理三个窗口事件
    # ===============================

    for vis in vis_list:

        if vis.poll_events():

            alive = True


        vis.update_renderer()



    # ===============================
    # 获取Prediction相机
    # ===============================

    try:

        master_view = (
            master_vis
            .get_view_control()
        )


        camera_param = (
            master_view
            .convert_to_pinhole_camera_parameters()
        )


        # 同步到Original

        for target_vis in [
            vis_original,
            vis_gt
        ]:

            target_view = (
                target_vis
                .get_view_control()
            )


            target_view\
                .convert_from_pinhole_camera_parameters(
                    camera_param,
                    allow_arbitrary=True
                )


    except Exception as e:

        pass



    if not alive:

        break



for vis in vis_list:

    vis.destroy_window()



for vis in vis_list:

    vis.destroy_window()






print("="*80)
print("Visualization finished")
print(output_dir)
print("="*80)