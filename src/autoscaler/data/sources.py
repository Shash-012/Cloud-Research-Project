"""Per-source settings for the training scripts (Phase 2 synthetic data or CloudSim Plus data, Step 3.15)."""
from dataclasses import dataclass

from ..config import load_config, merge_cloudsim
from ..schema import metrics_for
from .dataset import CLOUDSIM_TRANSITIONS, SYNTHETIC_TRANSITIONS


@dataclass
class Source:
    name: str
    cfg: dict
    data: str
    metrics: list
    group_col: str | None      # grouped CV / splits for correlated simulator rows
    suffix: str                # appended to table and figure names so sources never overwrite each other


def add_source_args(ap, default_model_dir="models"):
    ap.add_argument("--source", choices=["synthetic", "cloudsim"], default="synthetic")
    ap.add_argument("--cloudsim-config", default="configs/cloudsim.yaml")
    ap.add_argument("--model-dir", default=None,
                    help=f"default: {default_model_dir} (synthetic) or {default_model_dir}/cloudsim")


def resolve(args, default_model_dir="models") -> Source:
    proto = load_config(args.config)
    if args.source == "cloudsim":
        cfg = merge_cloudsim(proto, load_config(args.cloudsim_config))
        src = Source("cloudsim", cfg, str(CLOUDSIM_TRANSITIONS), metrics_for("cloudsim"), "run_id", "_cloudsim")
        args.model_dir = args.model_dir or f"{default_model_dir}/cloudsim"
    else:
        src = Source("synthetic", proto, str(SYNTHETIC_TRANSITIONS), metrics_for("synthetic"), None, "")
        args.model_dir = args.model_dir or default_model_dir
    if getattr(args, "data", None):
        src.data = args.data
    return src
