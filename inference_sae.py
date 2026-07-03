from inference_sae_common import build_arg_parser, run_sae_inference
from sae_utils import BioCLIPWithSAELayer


def parse_args():
    parser = build_arg_parser(
        "IP-Adapter Inference (BioCLIP SAE from checkpoint directory)",
        default_out_dir="outputs_bioclip_sae",
    )
    parser.set_defaults(sae_alpha=0.0)
    args = parser.parse_args()
    if not args.sae_dir:
        parser.error("--sae_dir is required and must point to a directory containing config.json and sae.pt")
    return args


if __name__ == "__main__":
    run_sae_inference(parse_args(), BioCLIPWithSAELayer, mode_name="layer_auto")
