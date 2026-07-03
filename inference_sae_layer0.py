from inference_sae_common import build_arg_parser, run_sae_inference
from sae_utils import BioCLIPWithSAELayer0


def parse_args():
    parser = build_arg_parser(
        "IP-Adapter Inference (BioCLIP SAE Layer 0)",
        default_out_dir="outputs_bioclip_sae_layer0",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_sae_inference(parse_args(), BioCLIPWithSAELayer0, mode_name="layer0")
