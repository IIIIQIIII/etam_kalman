import os
import torch
import numpy as np
import cv2
import datetime  # 新增时间模块

# ... [保持原有模型导入部分不变] ...
import matplotlib.pyplot as plt
import imageio

from efficient_track_anything.build_efficienttam import build_efficienttam_camera_predictor
from efficient_track_anything import get_checkpoint_path

# 初始化模型（根据实际路径修改）
checkpoint_path = "efficienttam_ti_512x512.pt"
checkpoint = get_checkpoint_path(checkpoint_path)
model_cfg = "configs/efficienttam/efficienttam_ti_512x512.yaml"
predictor = build_efficienttam_camera_predictor(model_cfg, checkpoint)

cap = cv2.VideoCapture(0)
if_init = False

# 用户点击点记录
clicked_points = []
clicked_labels = []

def mouse_callback(event, x, y, flags, param):
    global clicked_points, clicked_labels
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键正样本
        clicked_points.append([x, y])
        clicked_labels.append(1)
    elif event == cv2.EVENT_RBUTTONDOWN:  # 右键负样本
        clicked_points.append([x, y])
        clicked_labels.append(0)

cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("frame", mouse_callback)

# 初始化视频录制参数
is_recording = False
video_writer = None
output_filename = ""

# 预定义颜色表（BGR格式）保持不变...
# 预定义颜色表（BGR格式）
colors = [
    (255, 0, 0),   # 红
    (0, 255, 0),   # 绿
    (0, 0, 255),   # 蓝
    (0, 255, 255), # 黄
    (255, 0, 255), # 品红
    (255, 255, 0), # 青
]

# 在初始化代码后添加内存管理器
class MemoryManager:
    def __init__(self, max_cache=50):
        self.cache = {}
        self.max_cache = max_cache  # 最大缓存帧数
        
    def update(self, current_frame_idx):
        # 定期清理旧缓存
        if current_frame_idx % 10 == 0:
            outdated = [k for k in self.cache if k < current_frame_idx - self.max_cache]
            for key in outdated:
                del self.cache[key]
            torch.cuda.empty_cache()

# 初始化内存管理器
mem_manager = MemoryManager(max_cache=10)

# 在原有代码基础上增加录制功能
while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 录制状态提示
    if is_recording:
        cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)  # 红色状态灯

    if not if_init:
        # 初始化阶段
        predictor.load_first_frame(frame)
        ann_frame_idx = 0
        current_obj_id = 1  # 当前对象ID
        
        print("点击目标区域（左键正样本/右键负样本），按'n'确认当前对象，按'c'开始跟踪")
        
        while True:
            # 显示当前标注状态
            display_frame = frame.copy()
            # 绘制已有点
            for p, l in zip(clicked_points, clicked_labels):
                color = (0, 255, 0) if l == 1 else (0, 0, 255)
                cv2.circle(display_frame, tuple(p), 5, color, -1)
            # 显示提示信息
            cv2.putText(display_frame, f"Object {current_obj_id}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.imshow("frame", cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR))
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('n'):
                if clicked_points:
                    # 添加当前对象到跟踪器
                    points = np.array(clicked_points, dtype=np.float32)
                    labels = np.array(clicked_labels, dtype=np.int32)
                    predictor.add_new_prompt(
                        frame_idx=ann_frame_idx,
                        obj_id=current_obj_id,
                        points=points,
                        labels=labels
                    )
                    print(f"对象 {current_obj_id} 已添加")
                    current_obj_id += 1
                    clicked_points = []
                    clicked_labels = []
            elif key == ord('c'):
                if current_obj_id > 1:
                    if_init = True
                    break
                else:
                    print("请至少添加一个对象")
    # else:


    # 修改跟踪循环
    else:
        current_frame_idx = getattr(predictor, 'current_frame_idx', 0)
        mem_manager.update(current_frame_idx)
    
    # ...原有跟踪代码...
        # ... [保持原有跟踪逻辑不变] ...
        # 跟踪阶段
        obj_ids, mask_logits = predictor.track(frame)
        output = predictor.track(frame)
        # print("Track Output:", output)
        
        # 创建叠加层
        overlay = frame.copy()
        alpha = 0.6  # 透明度
        
        for idx, obj_id in enumerate(obj_ids):
            # 获取当前对象mask
            mask = (mask_logits[idx] > 0).squeeze().cpu().numpy().astype(np.uint8)
            color = colors[idx % len(colors)]
            
            # 绘制轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2)
            
            # 半透明填充
            mask_color = np.zeros_like(overlay)
            mask_color[mask > 0] = color
            overlay = cv2.addWeighted(overlay, 1, mask_color, 0.3, 0)
            
            # 计算中心点
            y, x = np.where(mask)
            if len(x) > 0:
                x_center, y_center = int(np.mean(x)), int(np.mean(y))
                cv2.circle(overlay, (x_center, y_center), 5, color, -1)
                cv2.putText(overlay, f'ID:{obj_id}', (x_center+5, y_center+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        
        # 融合叠加层
        frame = cv2.addWeighted(frame, 1-alpha, overlay, alpha, 0)

        # ==== 新增录制功能 ==== #
        if is_recording:
            if video_writer is None:
                # 自动生成带时间戳的文件名
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"tracking_{timestamp}.mp4"
                
                # 获取视频尺寸
                frame_height, frame_width = frame.shape[:2]
                
                # 自动选择可用编码器
                codecs = ['mp4v', 'XVID', 'MJPG', 'H264']
                for codec in codecs:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    if fourcc != -1:
                        break
                
                video_writer = cv2.VideoWriter(
                    output_filename,
                    fourcc,
                    25.0,  # 帧率
                    (frame_width, frame_height)
                )
                print(f"开始录制: {output_filename}")
            
            # 转换颜色空间并写入帧
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(bgr_frame)

    # 显示结果（保持原有逻辑）
    display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("frame", display_frame)

    # 键盘控制（新增录制控制）
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):  # 录制开关
        is_recording = not is_recording
        if not is_recording and video_writer is not None:
            video_writer.release()
            video_writer = None
            print(f"录制已保存: {output_filename}")
    elif key == ord('q'):
        break

# 释放资源
cap.release()
if video_writer is not None:
    video_writer.release()
cv2.destroyAllWindows()