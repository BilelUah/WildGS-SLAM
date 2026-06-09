import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
from typing import Dict, Tuple, Union

from thirdparty.depth_anything_v2.metric_depth.depth_anything_v2.dpt import (
    DepthAnythingV2,
)


def get_metric_depth_estimator(cfg: Dict) -> torch.nn.Module:
    """
    Get the metric depth estimator model based on the configuration.

    Args:
        cfg (Dict): Configuration dictionary.

    Returns:
        torch.nn.Module: The metric depth estimator model.
    """
    device = cfg["device"]
    depth_model = cfg["mono_prior"]["depth"]

    if "metric3d_vit" in depth_model:
        # Options: metric3d_vit_small, metric3d_vit_large, metric3d_vit_giant2
        model = torch.hub.load("yvanyin/metric3d", depth_model, pretrain=True)
    elif "dpt2" in depth_model:
        model = _create_dpt2_model(depth_model)
    else:
        # If use other metric depth estimator as prior, write the code here
        raise NotImplementedError("Unsupported depth model")
    return model.to(device).eval()


def _create_dpt2_model(depth_model: str) -> DepthAnythingV2:
    """
    Create a DPT2 model based on the depth model string.

    Args:
        depth_model (str): Depth model configuration string.

    Returns:
        DepthAnythingV2: Configured DPT2 model.
    """
    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {
            "encoder": "vitb",
            "features": 128,
            "out_channels": [96, 192, 384, 768],
        },
        "vitl": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
    }

    encoder, dataset, max_depth = depth_model.split("_")[1:4]
    config = {**model_configs[encoder], "max_depth": int(max_depth)}
    model = DepthAnythingV2(**config)

    weights_path = f"pretrained/depth_anything_v2_metric_{dataset}_{encoder}.pth"
    model.load_state_dict(
        torch.load(weights_path, map_location="cpu", weights_only=True)
    )

    return model


@torch.no_grad()
def predict_metric_depth(
    model: torch.nn.Module,
    idx: int,
    input_tensor: torch.Tensor,
    cfg: Dict,
    device: str,
    save_depth: bool = True,
) -> torch.Tensor:
    """
    Predict metric depth using the given model.

    Args:
        model (torch.nn.Module): The depth estimation model.
        idx (int): Image index.
        input_tensor (torch.Tensor): Input image tensor of shape (1, 3, H, W).
        cfg (Dict): Configuration dictionary.
        device (str): Device to run the model on.
        save_depth (bool): Whether to save the depth map.

    Returns:
        torch.Tensor: Predicted depth map.
    """
    depth_model = cfg["mono_prior"]["depth"]
    if "metric3d_vit" in depth_model:
        output = _predict_metric3d_depth(model, input_tensor, cfg, device)
    elif "dpt2" in depth_model:
        # dpt2 model takes np.uint8 as the dtype of input
        input_numpy = (255.0 * input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()).astype(
            np.uint8
        )
        depth = model.infer_image(input_numpy, input_size=518)
        output = torch.tensor(depth).to(device)
    else:
        # If use other metric depth estimator as prior, write the code here
        raise NotImplementedError("Unsupported depth model")

    _debug_metric_depth_io(
        input_tensor=input_tensor,
        depth_map=output,
        cfg=cfg,
        idx=idx
    )

    if save_depth:
        _save_depth_map(output, cfg, idx)

    return output


def _predict_metric3d_depth(
    model: torch.nn.Module, input_tensor: torch.Tensor, cfg: Dict, device: str
) -> torch.Tensor:
    # Refer from: https://github.com/YvanYin/Metric3D/blob/34afafe58d9543f13c01b65222255dab53333838/hubconf.py#L181
    image_size = (616, 1064)
    h, w = input_tensor.shape[-2:]
    scale = min(image_size[0] / h, image_size[1] / w)

    trans_totensor = transforms.Compose(
        [
            transforms.Resize((int(h * scale), int(w * scale))),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img_tensor = trans_totensor(input_tensor).to(device)

    pad_h, pad_w = image_size[0] - int(h * scale), image_size[1] - int(w * scale)
    pad_h_half, pad_w_half = pad_h // 2, pad_w // 2
    img_tensor = TF.pad(
        img_tensor,
        (pad_w_half, pad_h_half, pad_w - pad_w_half, pad_h - pad_h_half),
        padding_mode="constant",
        fill=0.0,
    )

    pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
    pred_depth, _, _ = model.inference({"input": img_tensor})
    pred_depth = pred_depth.squeeze()
    pred_depth = pred_depth[
        pad_info[0] : pred_depth.shape[0] - pad_info[1],
        pad_info[2] : pred_depth.shape[1] - pad_info[3],
    ]
    pred_depth = F.interpolate(
        pred_depth[None, None, :, :], (h, w), mode="bicubic"
    ).squeeze()

    canonical_to_real_scale = cfg["cam"]["fx"] / 1000.0

    if cfg.get("debug_metric_depth_scale", True):
        print("\n========== METRIC3D SCALE DEBUG ==========")
        print("[METRIC3D SCALE] cfg cam fx:", cfg["cam"]["fx"])
        print("[METRIC3D SCALE] cfg cam fy:", cfg["cam"].get("fy"))
        print("[METRIC3D SCALE] input h/w:", h, w)
        print("[METRIC3D SCALE] canonical_to_real_scale:", canonical_to_real_scale)
        print("==========================================\n")

    pred_depth = pred_depth * canonical_to_real_scale
    return torch.clamp(pred_depth, 0, 300)

def _debug_metric_depth_io(
    input_tensor: torch.Tensor,
    depth_map: torch.Tensor,
    cfg: Dict,
    idx: int,
) -> None:
    """
    Debug only.
    Saves the exact RGB tensor given to MetricDepth and the predicted depth.
    This does NOT modify input_tensor or depth_map.
    """
    debug_frames = {0, 9, 18, 98, 107, 134}

    idx_int = int(idx)
    if idx_int not in debug_frames:
        return

    try:
        import os
        import imageio.v2 as imageio

        scene = str(cfg.get("scene", "unknown_scene"))
        output_root = str(cfg.get("data", {}).get("output", "./output"))

        debug_dir = os.path.join(
            output_root,
            scene,
            "debug_metric_depth"
        )
        os.makedirs(debug_dir, exist_ok=True)

        # Save RGB input
        img = input_tensor.detach().float().cpu()

        # Expected shape: [1, 3, H, W]
        if img.ndim == 4:
            img = img[0]

        # [3, H, W] -> [H, W, 3]
        if img.ndim == 3 and img.shape[0] == 3:
            img = img.permute(1, 2, 0)

        img_np = img.numpy()
        img_u8 = (np.clip(img_np, 0.0, 1.0) * 255.0).astype(np.uint8)

        rgb_path = os.path.join(debug_dir, f"rgb_metric_input_{idx_int:06d}.png")
        imageio.imwrite(rgb_path, img_u8)

        # Save depth output
        d = depth_map.detach().float().cpu().numpy()
        depth_path = os.path.join(debug_dir, f"depth_metric_output_{idx_int:06d}.npy")
        np.save(depth_path, d)

        valid = d > 0

        print("\n========== METRIC DEPTH DEBUG ==========")
        print("[METRIC DEPTH DEBUG] frame:", idx_int)
        print("[METRIC DEPTH DEBUG] scene:", scene)
        print("[METRIC DEPTH DEBUG] rgb saved:", rgb_path)
        print("[METRIC DEPTH DEBUG] depth saved:", depth_path)
        print("[METRIC DEPTH DEBUG] input image shape:", img_u8.shape)
        print("[METRIC DEPTH DEBUG] input image mean:", float(img_u8.mean()))
        print("[METRIC DEPTH DEBUG] input image min/max:", int(img_u8.min()), int(img_u8.max()))
        print("[METRIC DEPTH DEBUG] depth shape:", d.shape)
        print("[METRIC DEPTH DEBUG] valid ratio:", float(valid.mean()))
        print("[METRIC DEPTH DEBUG] depth min:", float(d[valid].min()) if valid.any() else None)
        print("[METRIC DEPTH DEBUG] depth max:", float(d[valid].max()) if valid.any() else None)
        print("[METRIC DEPTH DEBUG] depth mean:", float(d[valid].mean()) if valid.any() else None)
        print("[METRIC DEPTH DEBUG] depth median:", float(np.median(d[valid])) if valid.any() else None)
        print("========================================\n")

    except Exception as e:
        print("[METRIC DEPTH DEBUG] failed:", repr(e))


def _save_depth_map(depth_map: torch.Tensor, cfg: Dict, idx: int) -> None:
    output_dir = f"{cfg['data']['output']}/{cfg['scene']}"
    output_path = f"{output_dir}/mono_priors/depths/{idx:05d}.npy"
    final_depth = depth_map.detach().cpu().float().numpy()
    np.save(output_path, final_depth)
