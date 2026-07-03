from inference_sae_common import build_arg_parser, run_sae_inference
from sae_utils import BioCLIPWithSAEPooled


def parse_args():
    parser = build_arg_parser(
        "IP-Adapter Inference (BioCLIP SAE Pooled)",
        default_out_dir="outputs_bioclip_sae_pooled",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_sae_inference(parse_args(), BioCLIPWithSAEPooled, mode_name="pooled")
