# Ported from MoMADiff utils/diffusion.py (create_gaussian_diffusion[/_ddim]),
# decoupled from the LDM factory and argparse namespace: parameters are
# explicit so Uni-Sign entrypoints can call it with plain values.
from diffusion_gen import gaussian_diffusion as gd
from diffusion_gen.respace import SpacedDiffusion, space_timesteps


def create_gaussian_diffusion(
    diffusion_steps: int = 1000,
    noise_schedule: str = "cosine",
    sigma_small: bool = True,
) -> SpacedDiffusion:
    """Full-step training diffusion (predict_xstart=True, MSE loss)."""
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    scale_beta = 1.0  # no scaling
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps, scale_beta)
    loss_type = gd.LossType.MSE

    timestep_respacing = [diffusion_steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def create_gaussian_diffusion_ddim(
    diffusion_steps: int = 1000,
    ddim_steps: int = 100,
    noise_schedule: str = "cosine",
    sigma_small: bool = True,
) -> SpacedDiffusion:
    """DDIM-respaced inference diffusion over the same base schedule."""
    predict_xstart = True
    scale_beta = 1.0
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps, scale_beta)
    loss_type = gd.LossType.MSE

    timestep_respacing = [ddim_steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )
