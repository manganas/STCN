from omegaconf import DictConfig, OmegaConf
import hydra


# def none_or_default(x, default):
#     return x if x is not None else default


@hydra.main(version_base=None, config_path="configs", config_name="base_config.yaml")
def parse_haparms(cfg):
    print(OmegaConf.to_yaml(cfg))

    hparams = OmegaConf.to_container(cfg)
    print(type(hparams))
    return hparams


# def check_hparams(hparams: dict):
#     if hparams["stage"] == 0:
#         # Static image pretraining
#         hparams["lr"] = none_or_default(hparams["lr"], 1e-5)
#         hparams["batch_size"] = none_or_default(hparams["batch_size"], 8)
#         hparams["iterations"] = none_or_default(hparams["iterations"], 300000)  # 300000
#         hparams["steps"] = none_or_default(hparams["steps"], [150000])
#         hparams["single_object"] = True
#     elif hparams["stage"] == 1:
#         # BL30K pretraining
#         hparams["lr"] = none_or_default(hparams["lr"], 1e-5)
#         hparams["batch_size"] = none_or_default(hparams["batch_size"], 4)
#         hparams["iterations"] = none_or_default(hparams["iterations"], 500000)
#         hparams["steps"] = none_or_default(hparams["steps"], [400000])
#         hparams["single_object"] = False
#     elif hparams["stage"] == 2:
#         # 300K main training for after BL30K
#         hparams["lr"] = none_or_default(hparams["lr"], 1e-5)
#         hparams["batch_size"] = none_or_default(hparams["batch_size"], 4)
#         hparams["iterations"] = none_or_default(hparams["iterations"], 300000)
#         hparams["steps"] = none_or_default(hparams["steps"], [250000])
#         hparams["single_object"] = False
#     elif hparams["stage"] == 3:
#         # 150K main training for after static image pretraining
#         hparams["lr"] = none_or_default(hparams["lr"], 1e-5)
#         hparams["batch_size"] = none_or_default(hparams["batch_size"], 4)
#         hparams["iterations"] = none_or_default(hparams["iterations"], 150000)
#         hparams["steps"] = none_or_default(hparams["steps"], [125000])
#         hparams["single_object"] = False

#     else:
#         raise NotImplementedError
