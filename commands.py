import hydra
from omegaconf import DictConfig

from cats_breed_detection.infer import infer
from cats_breed_detection.train import train
from cats_breed_detection.export_model import export_onnx


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(cfg)
    train(cfg)
    infer(cfg)
    export_onnx(cfg)


if __name__ == "__main__":
    main()
