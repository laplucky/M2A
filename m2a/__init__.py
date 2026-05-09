from .utils import (
    ensure_dir, cleanup_memory, print_memory_status,
    get_model_config_attr, get_layer_modules, set_strict_runtime,
)
from .data import (
    PreparedSample, read_json_samples, locate_segments, prepare_samples_unified,
    read_json_samples_recursive,
)
from .features import (
    collect_layer_features_with_hooks, compute_sampled_attention_rows_from_qk,
)
from .task_vectors import task_vectors_single_layer_unified
from .constraints import build_constraints_single_layer_unified
from .solvers_qk import (
    A_times_delta_qk_batched, AT_times_y_qk_batched,
    cg_single_head_batched, q_dense_project, k_dense_project,
)
from .solvers_vo import (
    A_times_delta_v, AT_times_y_v, cg_v, v_dense_project,
    A_times_delta_o, AT_times_y_o, cg_o, o_dense_project,
)
from .solvers_ffn import (
    A_times_delta_ffn_gate, AT_times_y_ffn_gate, cg_ffn_gate, ffn_gate_dense_project,
    A_times_delta_ffn_up, AT_times_y_ffn_up, cg_ffn_up, ffn_up_dense_project,
    A_times_delta_ffn_down, AT_times_y_ffn_down, cg_ffn_down, ffn_down_dense_project,
)
from .checkpoint import save_checkpoint, load_checkpoint, cleanup_checkpoint
from .metrics import (
    compute_frobenius_norm_layer, compute_cosine_similarity_layer,
    smooth_similarity_scores, compute_layer_selection_mask, compute_dynamic_alpha,
)
from .merge import M2A_merge
from .pipeline import optimized_layerwise_headwise_nullspace_projection
