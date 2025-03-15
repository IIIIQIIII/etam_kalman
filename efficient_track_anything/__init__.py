# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from hydra import initialize_config_module
from hydra.core.global_hydra import GlobalHydra

if not GlobalHydra.instance().is_initialized():
    initialize_config_module("efficient_track_anything", version_base="1.2")

import os

def get_project_root():
    """返回项目根目录"""
    # 假设包安装在site-packages中，向上两级是项目根目录
    return os.path.dirname(os.path.dirname(__file__))

def get_checkpoint_path(checkpoint_filename):
    """获取检查点文件路径"""
    root = get_project_root()
    return os.path.join(root, "checkpoints", checkpoint_filename)