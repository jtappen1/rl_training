# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""ONNX policy runner for evaluating exported/deployed policies (e.g. the deployed m20_policy.onnx)."""

from __future__ import annotations

import numpy as np
import torch


class OnnxPolicy:
    """Batched inference of a single-input, single-output ONNX MLP policy on CPU.

    Deployed exports often fix the batch dimension to 1; it is made dynamic here so all envs run in one call.
    """

    def __init__(self, path: str, obs_dim: int, num_actions: int):
        import onnx
        import onnxruntime as ort

        model = onnx.load(path)
        inputs = [i for i in model.graph.input if i.name not in {init.name for init in model.graph.initializer}]
        if len(inputs) != 1 or len(model.graph.output) != 1:
            raise ValueError(
                f"{path}: expected one input and one output (MLP policy), got inputs {[i.name for i in inputs]} "
                f"and outputs {[o.name for o in model.graph.output]}. Recurrent policies are not supported."
            )
        for vi in (inputs[0], model.graph.output[0]):
            vi.type.tensor_type.shape.dim[0].dim_param = "batch"
        del model.graph.value_info[:]  # stale intermediate shapes may pin the batch size

        in_dims = [d.dim_value for d in inputs[0].type.tensor_type.shape.dim[1:]]
        out_dims = [d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim[1:]]
        if in_dims != [obs_dim]:
            raise ValueError(f"{path}: policy input shape {in_dims} does not match the env observation size {obs_dim}")
        if out_dims != [num_actions]:
            raise ValueError(f"{path}: policy output shape {out_dims} does not match the env action size {num_actions}")

        self.session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        self.input_name = inputs[0].name

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.detach().to("cpu", dtype=torch.float32).numpy()
        (y,) = self.session.run(None, {self.input_name: x})
        return torch.from_numpy(np.asarray(y)).to(obs.device)
