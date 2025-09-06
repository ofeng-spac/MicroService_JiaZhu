#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主控程序 - 微服务架构控制器
负责协调多个机械臂的拍摄和处理任务
发送启动指令到相机服务
"""

import os
import time
import argparse
import yaml
import zenoh
import json
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger

# 导入现有工具模块
from zenoh_server.robot_client import control_robot_movement_sync


class MicroserviceMainController:
    """微服务架构主控制器"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = None
        self.zenoh_session = None
        self.timing_records = {}
        
    def load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"配置文件加载成功: {self.config_path}")
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            raise
    
    def init_zenoh(self):
        """初始化Zenoh会话"""
        try:
            self.zenoh_session = zenoh.open(zenoh.Config())
            logger.info("Zenoh会话初始化成功")
        except Exception as e:
            logger.error(f"Zenoh会话初始化失败: {e}")
            raise
    
    def send_capture_command(self, robot_ids: List[str], camera_params: Dict, capture_params: Dict):
        """发送拍摄指令到相机服务"""
        try:
            # 构建拍摄指令数据
            command_data = {
                "timestamp": time.time(),
                "robot_ids": robot_ids,
                "camera_params": camera_params,
                "capture_params": capture_params,
                "command_id": f"capture_{int(time.time())}"
            }
            
            # 序列化为JSON
            command_json = json.dumps(command_data, ensure_ascii=False)
            
            # 发送到Zenoh主题
            self.zenoh_session.put("jz/start_capture", command_json)
            logger.info(f"发送拍摄指令: {command_data['command_id']}")
            logger.debug(f"指令内容: {command_json}")
            
            return command_data['command_id']
            
        except Exception as e:
            logger.error(f"发送拍摄指令失败: {e}")
            raise
    
    def move_robots_to_capture_position(self, robot_ids: List[str], auto_run: bool = True):
        """移动机械臂到拍摄位置"""
        logger.info("开始移动机械臂到拍摄位置")
        start_time = time.time()
        
        try:
            # 获取机器人配置
            pot_key = next(iter(self.config['pots'].keys()))
            robot_config = self.config["robot"]
            
            for robot_id in robot_ids:
                logger.info(f"移动机械臂 {robot_id} 到拍摄位置")
                
                # 从robot_id中提取数字部分作为配置文件中的键
                # 例如: 'robot1' -> '1', 'robot2' -> '2'
                config_robot_id = robot_id.replace('robot', '')
                
                # 获取机器人初始位置
                robot_data = self.config['pots'][pot_key]['robots'][config_robot_id]
                matrix_robot_init = robot_data['source_pos_ini']
                robot_pos_ini = robot_data['robot_pos_ini']
                translation = robot_data['translation']
                
                # 调用机器人移动服务
                success = control_robot_movement_sync(
                    robot_config["ROBOT_IP"],
                    robot_config["ROBOT_PORT"],
                    robot_config["COMMAND"],
                    translation,  # robot_transformation_matrix
                    matrix_robot_init,  # robot_upper_pose
                    matrix_robot_init,  # matrix_robot_init
                    robot_pos_ini,  # robot_pos_ini
                    auto_run,  # auto_run
                    False,  # back
                    robot_config["joint_sequence_params"],
                    robot_config["move_sequence_params"]
                )
                
                if not success:
                    logger.warning(f"机械臂 {robot_id} 移动到拍摄位置失败")
                else:
                    logger.info(f"机械臂 {robot_id} 成功移动到拍摄位置")
            
            end_time = time.time()
            duration = end_time - start_time
            self.timing_records['move_to_capture'] = {
                'start': start_time,
                'end': end_time,
                'duration': duration
            }
            logger.info(f"所有机械臂移动到拍摄位置完成，耗时: {duration:.2f}秒")
            
        except Exception as e:
            logger.error(f"移动机械臂到拍摄位置失败: {e}")
            raise
    
    def wait_for_capture_completion(self, command_id: str, timeout: int = 60):
        """等待拍摄完成（可以通过监听响应主题实现）"""
        logger.info(f"等待拍摄任务完成: {command_id}")
        
        # 简单的等待实现，实际可以通过订阅响应主题来实现
        time.sleep(10)  # 假设拍摄需要10秒
        
        logger.info(f"拍摄任务完成: {command_id}")
    
    def move_robots_to_target_and_back(self, robot_ids: List[str], auto_run: bool = True):
        """移动机械臂到目标位置并返回"""
        logger.info("开始移动机械臂到目标位置")
        start_time = time.time()
        
        try:
            pot_key = next(iter(self.config['pots'].keys()))
            robot_config = self.config["robot"]
            
            for robot_id in robot_ids:
                logger.info(f"移动机械臂 {robot_id} 到目标位置")
                
                # 从robot_id中提取数字部分作为配置文件中的键
                config_robot_id = robot_id.replace('robot', '')
                
                robot_data = self.config['pots'][pot_key]['robots'][config_robot_id]
                translation = robot_data['translation']
                robot_pos_ini = robot_data['robot_pos_ini']
                
                # 移动到目标位置
                success = control_robot_movement_sync(
                    robot_config["ROBOT_IP"],
                    robot_config["ROBOT_PORT"],
                    robot_config["COMMAND"],
                    translation,  # robot_transformation_matrix
                    robot_pos_ini,  # robot_upper_pose
                    robot_pos_ini,  # matrix_robot_init
                    robot_pos_ini,  # robot_pos_ini
                    auto_run,  # auto_run
                    False,  # back
                    robot_config["joint_sequence_params"],
                    robot_config["move_sequence_params"]
                )
                
                if success:
                    logger.info(f"机械臂 {robot_id} 成功移动到目标位置")
                    time.sleep(2)
                    
                    # 返回初始位置
                    success = control_robot_movement_sync(
                        robot_config["ROBOT_IP"],
                        robot_config["ROBOT_PORT"],
                        robot_config["COMMAND"],
                        robot_pos_ini,  # robot_transformation_matrix
                        robot_pos_ini,  # robot_upper_pose
                        robot_pos_ini,  # matrix_robot_init
                        robot_pos_ini,  # robot_pos_ini
                        auto_run,  # auto_run
                        True,  # back (返回初始位置)
                        robot_config["joint_sequence_params"],
                        robot_config["move_sequence_params"]
                    )
                    
                    if success:
                        logger.info(f"机械臂 {robot_id} 成功返回初始位置")
                    else:
                        logger.warning(f"机械臂 {robot_id} 返回初始位置失败")
                else:
                    logger.warning(f"机械臂 {robot_id} 移动到目标位置失败")
            
            end_time = time.time()
            duration = end_time - start_time
            self.timing_records['move_tasks'] = {
                'start': start_time,
                'end': end_time,
                'duration': duration
            }
            logger.info(f"所有机械臂移动任务完成，耗时: {duration:.2f}秒")
            
        except Exception as e:
            logger.error(f"机械臂移动任务失败: {e}")
            raise
    
    def run_complete_workflow(self, robot_ids: List[str], args):
        """运行完整的工作流程"""
        start_total_time = time.time()
        self.timing_records['total'] = {'start': start_total_time}
        
        try:
            logger.info("开始执行完整工作流程")
            
            # 1. 移动机械臂到拍摄位置
            self.move_robots_to_capture_position(robot_ids, args.auto_run)
            
            # 2. 发送拍摄指令
            camera_params = {
                "camera_type": args.camera,
                "light_type": args.light,
                "exposure": args.exposure,
                "baseline": self.config.get("camera_params", {}).get("baseline", 0.074985),
                "focal_length": self.config.get("camera_params", {}).get("focal_length", 954.672)
            }
            
            capture_params = {
                "cut_method": args.cut,
                "tf_type": args.tf_type,
                "method": args.method,
                "auto_run": args.auto_run
            }
            
            command_id = self.send_capture_command(robot_ids, camera_params, capture_params)
            
            # 3. 等待拍摄完成
            self.wait_for_capture_completion(command_id)
            
            # 4. 移动机械臂到目标位置并返回
            self.move_robots_to_target_and_back(robot_ids, args.auto_run)
            
            # 记录总耗时
            end_total_time = time.time()
            self.timing_records['total']['end'] = end_total_time
            self.timing_records['total']['duration'] = end_total_time - start_total_time
            
            # 输出时间统计
            self.print_timing_summary()
            
            logger.info("完整工作流程执行成功")
            
        except Exception as e:
            logger.error(f"工作流程执行失败: {e}")
            raise
    
    def print_timing_summary(self):
        """输出时间统计"""
        logger.info("======= 程序执行时间统计 =======")
        logger.info(f"总程序耗时: {self.timing_records['total']['duration']:.2f}秒")
        
        if 'move_to_capture' in self.timing_records:
            logger.info(f"移动到拍摄位置: {self.timing_records['move_to_capture']['duration']:.2f}秒")
        
        if 'move_tasks' in self.timing_records:
            logger.info(f"移动任务: {self.timing_records['move_tasks']['duration']:.2f}秒")
        
        logger.info("===============================")
    
    def cleanup(self):
        """清理资源"""
        if self.zenoh_session:
            self.zenoh_session.close()
            logger.info("Zenoh会话已关闭")


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="微服务架构主控程序")
    parser.add_argument("--camera", type=str, default="OB", choices=["OB", "TY", "RS"],
                        help="相机类型 (OB, TY, RS)")
    parser.add_argument("--light", type=str, default="laser", choices=["laser", "flood"],
                        help="光源类型 (laser, flood)")
    parser.add_argument("--exposure", type=str, default="4000", help="曝光参数")
    parser.add_argument("--cut", type=str, default="box", choices=["box", "rectangle", "radius", ""],
                        help="点云截取方式 (box, rectangle, radius, 空字符串表示不截取)")
    parser.add_argument("--config", type=str, default="./config/robot_pot_init_1.yml", help="配置文件路径")
    parser.add_argument("--tf-type", type=str, default="rigid", choices=["rigid", "affine", "nonrigid", "nonrigid_constrained"],
                        help="变换类型")
    parser.add_argument("--method", type=str, default="filterreg", choices=["cpd", "filterreg", "gmmtree", "l2dist_regs"],
                        help="配准方法")
    parser.add_argument("--auto_run", action="store_true", help="自动运行，跳过所有用户交互和可视化")
    parser.add_argument("--robot_ids", type=str, nargs="+", default=["1"], help="机械臂ID列表")
    return parser.parse_args()


@logger.catch
def main():
    """主函数"""
    args = parse_args()
    
    # 配置日志
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, 'control.log')
    
    logger.add(
        log_file,
        rotation="10 MB",
        retention="7 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}"
    )
    
    controller = None
    
    try:
        logger.info("启动微服务架构主控程序")
        
        # 创建控制器
        controller = MicroserviceMainController(args.config)
        
        # 加载配置
        controller.load_config()
        
        # 初始化Zenoh
        controller.init_zenoh()
        
        # 处理robot_ids参数（支持逗号分隔的字符串）
        if len(args.robot_ids) == 1 and ',' in args.robot_ids[0]:
            robot_ids = args.robot_ids[0].split(',')
        else:
            robot_ids = args.robot_ids
        
        # 运行完整工作流程
        controller.run_complete_workflow(robot_ids, args)
        
        logger.info("主控程序执行完成")
        return 0
        
    except Exception as e:
        logger.error(f"主控程序执行失败: {e}")
        return 1
        
    finally:
        if controller:
            controller.cleanup()


if __name__ == "__main__":
    exit(main())