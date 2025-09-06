#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相机服务 - 微服务架构
负责接收拍摄指令，执行图像采集和处理，发布结果
订阅: jz/start_capture
发布: jz/capture/camera{i}/stereo_ir, jz/capture/camera{i}/processing_result
"""

import os
import time
import json
import zenoh
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from loguru import logger
import threading
import signal
import sys

# 导入现有模块
from zenoh_server.camera_client import take_photo
from zenoh_server.unimatch_client import run_unimatch_sync
from zenoh_server.depth_client import disparity_to_depth_sync, depth_to_pointcloud_sync
from zenoh_server.pointcloud_client import register_point_clouds_sync, compute_robot_transform_sync
from utils import load_pots_yml, load_eye_hand_matrix_choose


class CameraService:
    """相机服务类"""
    
    def __init__(self, config_path: str = "./config/robot_pot_init_1.yml"):
        self.config_path = config_path
        self.config = None
        self.zenoh_session = None
        self.subscriber = None
        self.publishers = {}
        self.running = False
        self.processing_lock = threading.Lock()
        
        # 预加载配置
        self.load_config()
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def load_config(self):
        """加载配置文件"""
        try:
            self.config = load_pots_yml(self.config_path)
            logger.info(f"配置文件加载成功: {self.config_path}")
            
            # 提取相机参数
            self.camera_params = self.config.get("camera_params", {})
            self.baseline = self.camera_params.get("baseline", 0.074985)
            self.focal_length = self.camera_params.get("focal_length", 954.672)
            
            # 提取其他配置
            self.ob_camera = self.config.get("OB_camera", "device0")
            self.eye_hand_matrix_path = self.config.get("eye_hand_matrix", "./eye_hand_matrix/eye_hand_matrix_1.json")
            self.intrinsics_fn = self.config.get("intrinsics_fn", "./structlight_intrinsics/structlight_self.yml")
            
            logger.info(f"相机设备: {self.ob_camera}")
            logger.info(f"相机参数 - 基线: {self.baseline}, 焦距: {self.focal_length}")
            
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            raise
    
    def init_zenoh(self):
        """初始化Zenoh会话"""
        try:
            self.zenoh_session = zenoh.open(zenoh.Config())
            logger.info("Zenoh会话初始化成功")
            
            # 创建订阅者
            self.subscriber = self.zenoh_session.declare_subscriber(
                "jz/start_capture",
                self._on_capture_command
            )
            logger.info("订阅 jz/start_capture 成功")
            
        except Exception as e:
            logger.error(f"Zenoh会话初始化失败: {e}")
            raise
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        logger.info(f"接收到信号 {signum}，正在关闭服务...")
        self.stop()
        sys.exit(0)
    
    def _on_capture_command(self, sample):
        """处理拍摄指令"""
        try:
            # 解析指令数据
            command_data = json.loads(sample.payload.decode('utf-8'))
            logger.info(f"接收到拍摄指令: {command_data.get('command_id', 'unknown')}")
            logger.debug(f"指令内容: {command_data}")
            
            # 在新线程中处理，避免阻塞订阅者
            processing_thread = threading.Thread(
                target=self._process_capture_command,
                args=(command_data,)
            )
            processing_thread.start()
            
        except Exception as e:
            logger.error(f"处理拍摄指令失败: {e}")
    
    def _process_capture_command(self, command_data: Dict[str, Any]):
        """处理拍摄指令的具体逻辑"""
        with self.processing_lock:
            try:
                start_time = time.time()
                command_id = command_data.get('command_id', 'unknown')
                robot_ids = command_data.get('robot_ids', ['1'])
                camera_params = command_data.get('camera_params', {})
                capture_params = command_data.get('capture_params', {})
                
                logger.info(f"开始处理拍摄任务: {command_id}")
                
                # 执行图像采集和处理流程
                results = self._execute_capture_and_processing(
                    robot_ids, camera_params, capture_params
                )
                
                # 发布处理结果
                self._publish_results(command_id, robot_ids, results)
                
                end_time = time.time()
                duration = end_time - start_time
                logger.info(f"拍摄任务完成: {command_id}, 耗时: {duration:.2f}秒")
                
            except Exception as e:
                logger.error(f"拍摄任务处理失败: {e}")
                # 发布错误信息
                self._publish_error(command_data.get('command_id', 'unknown'), str(e))
    
    def _execute_capture_and_processing(self, robot_ids: List[str], camera_params: Dict, capture_params: Dict) -> Dict:
        """执行图像采集和处理"""
        results = {}
        
        try:
            # 1. 拍摄图像
            logger.info("开始拍摄图像")
            capture_start_time = time.time()
            
            camera_type = camera_params.get('camera_type', 'OB')
            light_type = camera_params.get('light_type', 'laser')
            exposure = camera_params.get('exposure', '4000')
            
            # 调用拍摄服务
            photo_results = take_photo(
                camera_type=camera_type,
                light_type=light_type,
                exposure=exposure,
                device_id=self.ob_camera
            )
            
            capture_end_time = time.time()
            logger.info(f"图像拍摄完成，耗时: {capture_end_time - capture_start_time:.2f}秒")
            
            # 发布原始图像数据
            for i, robot_id in enumerate(robot_ids):
                self._publish_stereo_images(i, photo_results)
            
            # 2. 处理每个机械臂的数据
            for robot_id in robot_ids:
                logger.info(f"开始处理机械臂 {robot_id} 的数据")
                robot_result = self._process_robot_data(
                    robot_id, photo_results, camera_params, capture_params
                )
                results[robot_id] = robot_result
            
            return results
            
        except Exception as e:
            logger.error(f"图像采集和处理失败: {e}")
            raise
    
    def _process_robot_data(self, robot_id: str, photo_results: Dict, camera_params: Dict, capture_params: Dict) -> Dict:
        """处理单个机械臂的数据"""
        try:
            device_id = f"device{int(robot_id)-1}"
            
            # 1. 计算视差图
            logger.info(f"机械臂 {robot_id}: 开始计算视差图")
            disparity_start_time = time.time()
            
            disparity_result = run_unimatch_sync(
                left_image_path=photo_results.get('left_image_path'),
                right_image_path=photo_results.get('right_image_path'),
                device_id=device_id
            )
            
            disparity_end_time = time.time()
            logger.info(f"机械臂 {robot_id}: 视差图计算完成，耗时: {disparity_end_time - disparity_start_time:.2f}秒")
            
            # 2. 转换深度图
            logger.info(f"机械臂 {robot_id}: 开始转换深度图")
            depth_start_time = time.time()
            
            depth_result = disparity_to_depth_sync(
                disparity_path=disparity_result.get('disparity_path'),
                baseline=camera_params.get('baseline', self.baseline),
                focal_length=camera_params.get('focal_length', self.focal_length),
                device_id=device_id
            )
            
            depth_end_time = time.time()
            logger.info(f"机械臂 {robot_id}: 深度图转换完成，耗时: {depth_end_time - depth_start_time:.2f}秒")
            
            # 3. 转换点云
            logger.info(f"机械臂 {robot_id}: 开始转换点云")
            pointcloud_start_time = time.time()
            
            pointcloud_result = depth_to_pointcloud_sync(
                depth_path=depth_result.get('depth_path'),
                color_image_path=photo_results.get('left_image_path'),
                intrinsics_path=self.intrinsics_fn,
                device_id=device_id
            )
            
            pointcloud_end_time = time.time()
            logger.info(f"机械臂 {robot_id}: 点云转换完成，耗时: {pointcloud_end_time - pointcloud_start_time:.2f}秒")
            
            # 4. 点云配准
            logger.info(f"机械臂 {robot_id}: 开始点云配准")
            registration_start_time = time.time()
            
            # 获取源点云路径
            pot_key = next(iter(self.config['pots'].keys()))
            source_down_path = self.config['pots'][pot_key]['source_down_fn']
            
            registration_result = register_point_clouds_sync(
                source_path=source_down_path,
                target_path=pointcloud_result.get('pointcloud_path'),
                method=capture_params.get('method', 'filterreg'),
                tf_type=capture_params.get('tf_type', 'rigid'),
                cut_method=capture_params.get('cut_method', 'box'),
                device_id=device_id
            )
            
            registration_end_time = time.time()
            logger.info(f"机械臂 {robot_id}: 点云配准完成，耗时: {registration_end_time - registration_start_time:.2f}秒")
            
            # 5. 计算机器人变换矩阵
            logger.info(f"机械臂 {robot_id}: 开始计算机器人变换矩阵")
            transform_start_time = time.time()
            
            transform_result = compute_robot_transform_sync(
                registration_matrix=registration_result.get('transformation_matrix'),
                eye_hand_matrix_path=self.eye_hand_matrix_path,
                device_id=device_id
            )
            
            transform_end_time = time.time()
            logger.info(f"机械臂 {robot_id}: 机器人变换矩阵计算完成，耗时: {transform_end_time - transform_start_time:.2f}秒")
            
            # 整合结果
            result = {
                'robot_id': robot_id,
                'device_id': device_id,
                'disparity_result': disparity_result,
                'depth_result': depth_result,
                'pointcloud_result': pointcloud_result,
                'registration_result': registration_result,
                'transform_result': transform_result,
                'timing': {
                    'disparity_duration': disparity_end_time - disparity_start_time,
                    'depth_duration': depth_end_time - depth_start_time,
                    'pointcloud_duration': pointcloud_end_time - pointcloud_start_time,
                    'registration_duration': registration_end_time - registration_start_time,
                    'transform_duration': transform_end_time - transform_start_time
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"机械臂 {robot_id} 数据处理失败: {e}")
            raise
    
    def _publish_stereo_images(self, camera_index: int, photo_results: Dict):
        """发布双目图像数据"""
        try:
            topic = f"jz/capture/camera{camera_index}/stereo_ir"
            
            # 构建图像数据
            image_data = {
                'timestamp': time.time(),
                'camera_index': camera_index,
                'left_image_path': photo_results.get('left_image_path'),
                'right_image_path': photo_results.get('right_image_path'),
                'ir_image_path': photo_results.get('ir_image_path'),
                'metadata': {
                    'baseline': self.baseline,
                    'focal_length': self.focal_length,
                    'device_id': self.ob_camera
                }
            }
            
            # 发布数据
            image_json = json.dumps(image_data, ensure_ascii=False)
            self.zenoh_session.put(topic, image_json)
            
            logger.info(f"发布图像数据到: {topic}")
            
        except Exception as e:
            logger.error(f"发布图像数据失败: {e}")
    
    def _publish_results(self, command_id: str, robot_ids: List[str], results: Dict):
        """发布处理结果"""
        try:
            for robot_id in robot_ids:
                camera_index = int(robot_id) - 1
                topic = f"jz/capture/camera{camera_index}/processing_result"
                
                result_data = {
                    'command_id': command_id,
                    'timestamp': time.time(),
                    'robot_id': robot_id,
                    'camera_index': camera_index,
                    'status': 'success',
                    'result': results.get(robot_id, {})
                }
                
                result_json = json.dumps(result_data, ensure_ascii=False, default=str)
                self.zenoh_session.put(topic, result_json)
                
                logger.info(f"发布处理结果到: {topic}")
            
            # 发布总体完成状态
            completion_topic = "jz/capture/completion"
            completion_data = {
                'command_id': command_id,
                'timestamp': time.time(),
                'status': 'completed',
                'robot_ids': robot_ids,
                'total_robots': len(robot_ids)
            }
            
            completion_json = json.dumps(completion_data, ensure_ascii=False)
            self.zenoh_session.put(completion_topic, completion_json)
            
            logger.info(f"发布完成状态到: {completion_topic}")
            
        except Exception as e:
            logger.error(f"发布处理结果失败: {e}")
    
    def _publish_error(self, command_id: str, error_message: str):
        """发布错误信息"""
        try:
            error_topic = "jz/capture/error"
            error_data = {
                'command_id': command_id,
                'timestamp': time.time(),
                'status': 'error',
                'error_message': error_message
            }
            
            error_json = json.dumps(error_data, ensure_ascii=False)
            self.zenoh_session.put(error_topic, error_json)
            
            logger.error(f"发布错误信息到: {error_topic}")
            
        except Exception as e:
            logger.error(f"发布错误信息失败: {e}")
    
    def start(self):
        """启动服务"""
        try:
            logger.info("启动相机服务")
            
            # 初始化Zenoh
            self.init_zenoh()
            
            self.running = True
            logger.info("相机服务启动成功，等待拍摄指令...")
            
            # 保持服务运行
            while self.running:
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"相机服务启动失败: {e}")
            raise
    
    def stop(self):
        """停止服务"""
        logger.info("正在停止相机服务...")
        self.running = False
        
        if self.subscriber:
            self.subscriber.undeclare()
            
        if self.zenoh_session:
            self.zenoh_session.close()
            
        logger.info("相机服务已停止")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="相机服务 - 微服务架构")
    parser.add_argument("--config", type=str, default="./config/robot_pot_init_1.yml", help="配置文件路径")
    args = parser.parse_args()
    
    # 配置日志
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, 'camera_service.log')
    
    logger.add(
        log_file,
        rotation="10 MB",
        retention="7 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}"
    )
    
    service = None
    
    try:
        logger.info("初始化相机服务")
        service = CameraService(args.config)
        service.start()
        
    except KeyboardInterrupt:
        logger.info("接收到中断信号")
        
    except Exception as e:
        logger.error(f"相机服务运行失败: {e}")
        return 1
        
    finally:
        if service:
            service.stop()
    
    return 0


if __name__ == "__main__":
    exit(main())