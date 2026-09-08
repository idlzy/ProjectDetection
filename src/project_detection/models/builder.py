from .detectors import FCOS3D


def build_model(config):
    model = config["model"]
    data = config["data"]
    return FCOS3D(
        num_classes=len(data["classes"]),
        backbone=model.get("backbone", "efficientnet_b0_hat_compatible"),
        neck=model.get("neck", "bifpn"),
        neck_channels=model["neck_channels"],
        bifpn_stacks=model["bifpn_stacks"],
        head_channels=model["head_channels"],
        stacked_convs=model["stacked_convs"],
        num_attrs=model["num_attrs"],
        probabilistic_depth=model["probabilistic_depth"],
        geometric_depth=model.get("geometric_depth", False),
        depth_bin_unit=model["depth_bin_unit"],
        depth_bin_max=model["depth_bin_max"],
        backbone_dcn_stages=model.get(
            "backbone_dcn_stages", (False, False, False, False)
        ),
        backbone_frozen_stages=model.get("backbone_frozen_stages", -1),
        backbone_norm_eval=model.get("backbone_norm_eval", False),
        head_norm=model.get("head_norm", "batch"),
        dcn_on_last_conv=model.get("dcn_on_last_conv", False),
        deform_groups=model.get("deform_groups", 1),
    )
