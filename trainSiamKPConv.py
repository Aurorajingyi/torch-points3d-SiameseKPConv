import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from torch_points3d.trainer_SiamKPConv import Trainer
import os
# # Set CUDA_LAUNCH_BLOCKING=1 for synchronous execution
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# @hydra.main(config_path="conf/configUrb3D.yaml")
@hydra.main(config_path="/mnt/d/n-siamkpconv/torch-points3d-SiameseKPConv/conf", config_name="configSiamKPConv.yaml")
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
    if cfg.pretty_print:
        print(cfg.pretty())
    trainer = Trainer(cfg)
    trainer.train()
    # trainer.eval()
    # https://github.com/facebookresearch/hydra/issues/440
    GlobalHydra.get_state().clear()
    return 0


if __name__ == "__main__":
    main()
