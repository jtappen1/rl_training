# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""D435i depth pipeline shared by the simulator and the robot (see docs/depth_pipeline.md).

`d435i.py` and `preprocess.py` have no Isaac Lab dependency and are what the robot-side node uses;
`stereo_noise.py` is simulator-only.
"""

from .d435i import D435I, LITE3_D435I_MOUNT, M20_D435I_MOUNT, CameraMount, D435iSpec
from .preprocess import normalize, pool_depth, preprocess, sanitize
