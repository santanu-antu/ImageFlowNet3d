# 3D model creation utilities.
# Adapted from script_util.py for true volumetric 3D processing.

from .unet_3d import UNetModel3D

NUM_CLASSES = 1000


def create_model_3d(
    volume_size,
    num_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
    in_channels=1,
    **kwargs,
):
    """
    Create a 3D UNet model for volumetric data.

    :param volume_size: the size of the input volume (assumes cubic: D=H=W).
    :param num_channels: base channel count for the model.
    :param num_res_blocks: number of residual blocks per downsample.
    :param channel_mult: channel multiplier for each level of the UNet.
                         If empty string, will be set based on volume_size.
    :param learn_sigma: if True, output channels are doubled for sigma prediction.
    :param class_cond: if True, model is class-conditional.
    :param use_checkpoint: if True, use gradient checkpointing.
    :param attention_resolutions: comma-separated list of resolutions for attention.
    :param num_heads: number of attention heads.
    :param num_head_channels: if specified, overrides num_heads.
    :param num_heads_upsample: number of attention heads for upsampling path.
    :param use_scale_shift_norm: use FiLM-like conditioning.
    :param dropout: dropout probability.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_fp16: use float16 precision.
    :param use_new_attention_order: use different attention pattern.
    :param in_channels: number of input channels.
    :return: UNetModel3D instance.
    """
    if channel_mult == "":
        # Default channel multipliers for different volume sizes
        # Using fewer levels than 2D to manage memory for 3D volumes
        if volume_size == 128:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif volume_size == 64:
            channel_mult = (1, 2, 4, 8)
        elif volume_size == 32:
            channel_mult = (1, 2, 4)
        elif volume_size == 16:
            channel_mult = (1, 2)
        else:
            raise ValueError(f"unsupported volume size: {volume_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    if attention_resolutions:
        for res in attention_resolutions.split(","):
            if res:
                attention_ds.append(volume_size // int(res))

    return UNetModel3D(
        volume_size=volume_size,
        in_channels=in_channels,
        model_channels=num_channels,
        out_channels=(in_channels if not learn_sigma else 2 * in_channels),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )
