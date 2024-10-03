from pathlib import Path

import torch

from cats_breed_detection.model import CatModel


def export_onnx(cfg):
    model = CatModel.load_from_checkpoint(checkpoint_path=cfg.model.save_model_name)

    model_name = f"{cfg.export.export_name}.onnx"

    filepath = Path(cfg.export.export_path) / model_name
    filepath.parent.mkdir(parents=True, exist_ok=True)

    input_sample = torch.randn(tuple(cfg.export.input_sample_shape))

    model.to_onnx(
        filepath,
        input_sample,
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=["IMAGES"],
        output_names=["CLASS_PROBS"],
        dynamic_axes={
            "IMAGES": {0: "BATCH_SIZE"},
            "CLASS_PROBS": {0: "BATCH_SIZE"},
        },
    )
