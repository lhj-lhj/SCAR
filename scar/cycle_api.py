"""Cycle-training API: single entry point for all cycle pipeline imports."""

from __future__ import annotations

from .config import (
    align_lvp_action_dim,
    align_idm_seq_len,
    build_lvp_cfg,
    build_idm_cfg,
    get_lvp_target_seq_len,
    infer_idm_image_hw,
    parse_config_args,
    parse_comma_separated_list,
    resolve_path,
)
from .environment import (
    CYCLE_TRAIN_SCRIPT_PATH,
    DEFAULT_LIBERO_NEGATIVE_PROMPT_EMBED,
    DEFAULT_LIBERO_PROMPT,
    DEFAULT_LIBERO_PROMPT_EMBED,
    LVP_ROOT,
    SCAR_ROOT,
)
from .evaluation import (
    all_gather_object_payload,
    build_sampling_batch,
    compose_lvp_video_prediction,
    infer_obs_shape,
    rollout_lvp_latent_trainable,
    sample_lvp_video,
    save_eval_videos,
    video_tensor_to_uint8_numpy,
)
from .models import (
    LatentActionToGTActionLinearHead,
    LatentActionToGTActionTransformerHead,
    LatentSpaceIDM,
    build_gt_action_head,
    build_gt_action_padding_mask,
    build_local_timesteps,
    describe_gt_action_head,
    prepare_latent_idm_inputs,
    resolve_gt_action_transformer_nheads,
)
from .objectives import (
    ObjectiveOutput,
    build_lvp_condition_latents,
    build_wrong_z_permutation,
    compute_cycle_objectives,
    compute_future_flow_mse_distance,
    compute_gt_action_aux_loss,
    compute_lvp_cycle_loss,
    compute_lvp_recon_loss_from_video_lat,
    compute_wrong_z_loss,
    encode_lvp_video_latents,
    forward_lvp_flow,
    get_cycle_loss_scale,
    get_gt_action_loss_scale,
    predict_lvp_clean_video_lat_single_step,
    project_action_tokens_to_latent_bias,
    resolve_conditioning_action_dim,
    resolve_conditioning_actions,
    run_idm_on_video_lat,
    should_compute_eval_cycle_loss,
    should_compute_wrong_z_loss,
    unpack_lvp_action_encoder_output,
)
from .runtime import (
    DistributedContext,
    LatentActionIDMWrapper,
    build_joint_optimizer,
    build_lr_scheduler,
    build_lvp_prior,
    build_prompt_context,
    cast_tensor_tree,
    cleanup_distributed,
    debug_log_once,
    distributed_barrier,
    freeze_module,
    get_current_optimizer_lrs,
    get_lvp_module_map,
    get_lvp_optimizer_hparams,
    init_wandb_run,
    iter_unique_trainable_params,
    load_prompt_embedding,
    maybe_ddp_no_sync,
    maybe_wrap_ddp,
    reduce_pair_mean,
    reduce_scalar,
    resolve_lvp_trainable_modules,
    save_checkpoint,
    select_rgb_channels,
    set_lvp_mode,
    set_seed,
    setup_distributed,
    take_batch_prefix,
    temporarily_disable_grads,
    temporarily_frozen_eval,
    temporarily_unwrap_lvp_ddp,
    to_lvp_range,
    trim_batch,
    unfreeze_module,
    unwrap_idm_module,
    unwrap_module,
)

SCRIPT_PATH = CYCLE_TRAIN_SCRIPT_PATH
parse_args = parse_config_args


def train(args):
    from .engines import train_cycle

    return train_cycle(args)


def main() -> None:
    from .engines import main_cycle

    main_cycle()
