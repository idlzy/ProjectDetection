"""Shared raw-head ONNX output contract for export and inference."""

BASE_OUTPUT_FIELDS = (
    "cls", "bbox", "direction", "attribute", "centerness",
)


def output_fields(config):
    fields = list(BASE_OUTPUT_FIELDS)
    if config["model"]["probabilistic_depth"]:
        fields.append("depth_logits")
    if config["model"].get("geometric_depth", False):
        fields.append("geo_weight")
    return fields


def output_names(config):
    return [
        "p%d_%s" % (level, field)
        for level in range(3, 8)
        for field in output_fields(config)
    ]
