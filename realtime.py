from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
import json
import time
import open3d as o3d
from typing import Dict, Optional

from wilor.models import WiLoR, load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from wilor.utils.renderer import Renderer, cam_crop_to_full
from ultralytics import YOLO

LIGHT_PURPLE=(0.25098039,  0.274117647,  0.65882353)


def project_full_img(points, cam_trans, focal_length, img_res): 
    camera_center = [img_res[0] / 2., img_res[1] / 2.]
    K = torch.eye(3) 
    K[0,0] = focal_length
    K[1,1] = focal_length
    K[0,2] = camera_center[0]
    K[1,2] = camera_center[1]
    points = points + cam_trans
    points = points / points[..., -1:] 
    
    V_2d = (K @ points.T).T 
    return V_2d[..., :-1]



def main():
    # Download and load checkpoints
    model, model_cfg = load_wilor(checkpoint_path = './pretrained_models/wilor_final.ckpt' , cfg_path= './pretrained_models/model_config.yaml')
    detector = YOLO('./pretrained_models/detector.pt')
    # Setup the renderer
    # renderer = Renderer(model_cfg, faces=model.mano.faces)
    # renderer_side = Renderer(model_cfg, faces=model.mano.faces)

    device   = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model    = model.to(device)
    detector = detector.to(device)
    model.eval()
    print("Model and detector on device:", device)

    cap = cv2.VideoCapture('/dev/video0')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("Cannot open /dev/video0")
        return

    skeleton = [
        [0,1],[1,2],[2,3],[3,4],      # 拇指
        [0,5],[5,6],[6,7],[7,8],      # 食指
        [0,9],[9,10],[10,11],[11,12], # 中指
        [0,13],[13,14],[14,15],[15,16], # 无名指
        [0,17],[17,18],[18,19],[19,20]  # 小指
    ]

    # vis = o3d.visualization.Visualizer()
    # vis.create_window(window_name="WiLoR 3D Hand Keypoints", width=1600, height=1200)
    # joints_pcd = o3d.geometry.PointCloud()
    # verts_pcd = o3d.geometry.PointCloud()
    # line_set = o3d.geometry.LineSet()
    # added_flag = False
    # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

    # ctr = vis.get_view_control()
    # ctr.set_front([0, 0, -1])      # 摄像机朝向 z+
    # ctr.set_lookat([0, 0, 0])     # 视点在原点
    # ctr.set_up([0, -1, 0])         # y 轴向上
    # ctr.set_zoom(0.7)             # 可根据实际场景调整缩放

    try:
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            detections = detector(frame, conf=0.3, verbose=False)[0]
            bboxes, is_right = [], []
            for det in detections:
                Bbox = det.boxes.data.cpu().detach().squeeze().numpy()
                is_right.append(det.boxes.cls.cpu().detach().squeeze().item())
                bboxes.append(Bbox[:4].tolist())

            if len(bboxes) == 0:
                elapsed = time.time() - start_time
                fps = 1.0 / elapsed if elapsed > 0 else 0
                print("FPS: {:.2f}".format(fps))
            #     cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            #     big_image = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LINEAR)
            #     cv2.imshow('Processed Image', big_image)
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         break
                continue
            
            processed_image = frame.copy()
            boxes = np.stack(bboxes)
            right = np.stack(is_right)
            dataset = ViTDetDataset(model_cfg, frame, boxes, right, rescale_factor=2.0)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

            # all_verts = []
            # all_cam_t = []
            # all_right = []
            # all_joints= []
            # all_kpts  = []

            for batch in dataloader: 
                batch = recursive_to(batch, device)
                with torch.no_grad():
                    out = model(batch) 

                multiplier    = (2*batch['right']-1)
                pred_cam      = out['pred_cam']
                pred_cam[:,1] = multiplier*pred_cam[:,1]
                box_center    = batch["box_center"].float()
                box_size      = batch["box_size"].float()
                img_size      = batch["img_size"].float()
                scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
                pred_cam_t_full     = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

                batch_size = batch['img'].shape[0]
                all_points = []
                all_lines = []
                all_verts = []
                offset = 0
                for n in range(batch_size):
                    keypoints_3d = out['pred_keypoints_3d'][n].detach().cpu().numpy()  # (21, 3)
                    verts = out['pred_vertices'][n].detach().cpu().numpy()  # (778, 3)
                    is_right = batch['right'][n].cpu().numpy()
                    cam_t = pred_cam_t_full[n]
                    keypoints_3d[:,0] = (2*is_right-1)*keypoints_3d[:,0]
                    verts[:,0] = (2*is_right-1)*verts[:,0]


                    # wrist点坐标
                    wrist_xyz = keypoints_3d[0].copy()
                    # 所有点坐标减去wrist的坐标，实现左右手空间完全对齐
                    # keypoints_3d += wrist_xyz
                    # verts += wrist_xyz

                    keypoints_3d_world = keypoints_3d + cam_t
                    verts_world = verts + cam_t

                    all_points.append(keypoints_3d_world)
                    all_lines += [[i+offset, j+offset] for i, j in skeleton]
                    all_verts.append(verts_world)
                    offset += keypoints_3d_world.shape[0]

                    wrist = keypoints_3d_world[0]
                    print()
                    print(f"Wrist 3D coordinates: [{wrist[0]:.2f}, {wrist[1]:.2f}, {wrist[2]:.2f}]")



                # # 可视化所有关键点和骨架
                # if all_points and all_verts and all_lines:
                #     all_points = np.concatenate(all_points, axis=0)
                #     joints_pcd.points = o3d.utility.Vector3dVector(all_points)
                #     joints_pcd.colors = o3d.utility.Vector3dVector(np.tile([[0.2, 0.7, 0.9]], (all_points.shape[0], 1)))

                #     all_verts = np.concatenate(all_verts, axis=0)
                #     verts_pcd.points = o3d.utility.Vector3dVector(all_verts)
                #     verts_pcd.colors = o3d.utility.Vector3dVector(np.tile([[0.8, 0.8, 0.8]], (all_verts.shape[0], 1)))

                #     line_set.points = o3d.utility.Vector3dVector(all_points)
                #     line_set.lines = o3d.utility.Vector2iVector(all_lines)
                #     line_set.colors = o3d.utility.Vector3dVector([[1,0,0]]*len(all_lines))

                    # # 可视化所有网格
                    # if not added_flag:
                    #     vis.add_geometry(joints_pcd)
                    #     vis.add_geometry(verts_pcd)
                    #     vis.add_geometry(line_set)
                    #     vis.add_geometry(coord_frame)
                    #     added_flag = True
                    # else:
                    #     vis.update_geometry(joints_pcd)
                    #     vis.update_geometry(verts_pcd)
                    #     vis.update_geometry(line_set)
                    #     vis.update_geometry(coord_frame)
                    # vis.poll_events()
                    # vis.update_renderer()
                
                # batch_size = batch['img'].shape[0]
                # for n in range(batch_size):

                #     verts = out['pred_vertices'][n].detach().cpu().numpy()
                #     joints = out['pred_keypoints_3d'][n].detach().cpu().numpy()

                #     is_right = batch['right'][n].cpu().numpy()
                #     verts[:, 0] = (2 * is_right - 1) * verts[:, 0]
                #     joints[:, 0] = (2 * is_right - 1) * joints[:, 0]
                #     cam_t = pred_cam_t_full[n]
                #     kpts_2d = project_full_img(joints, cam_t, scaled_focal_length, img_size[n])
                #     kpts_2d = kpts_2d.cpu().numpy().astype(np.int32)

                #     # 绘制骨架
                #     for i, j in skeleton:
                #         pt1 = tuple(kpts_2d[i])
                #         pt2 = tuple(kpts_2d[j])
                #         cv2.line(processed_image, pt1, pt2, (0,255,0), 2)
                #     for idx, pt in enumerate(kpts_2d):
                #         color = (0,0,255) if idx==0 else (255,0,0)
                #         cv2.circle(processed_image, tuple(pt), 2, color, -1)
                
                    # all_verts.append(verts)
                    # all_cam_t.append(cam_t)
                    # all_right.append(is_right)
                    # all_joints.append(joints)
                    # all_kpts.append(kpts_2d)
                    
            # # Render front view
            # if len(all_verts) > 0:
            #     misc_args = dict(
            #         mesh_base_color=LIGHT_PURPLE,
            #         scene_bg_color=(1, 1, 1),
            #         focal_length=scaled_focal_length,
            #     )
            #     cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)

            #     # Overlay image
            #     input_img = frame.astype(np.float32)[:,:,::-1]/255.0
            #     input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
            #     processed_image = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]
            
            elapsed = time.time() - start_time
            fps = 1.0 / elapsed if elapsed > 0 else 0
            print("FPS: {:.2f}".format(fps))
            # cv2.putText(processed_image, f"FPS: {fps:.2f}", (20, 40),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # big_image = cv2.resize(processed_image, (1920, 1080), interpolation=cv2.INTER_LINEAR)
            # cv2.imshow('Processed Image', big_image)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
