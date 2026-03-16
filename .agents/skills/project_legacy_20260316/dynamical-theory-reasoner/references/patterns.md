# Pattern Mapping

## Markov closure

- `lagged_rmse << markov_rmse`: the coordinate is not close to Markov; memory or unresolved latent variables still matter.
- `lagged_rmse ~= markov_rmse`: the coordinate is closer to a closed state representation.

## Spectral preservation

- Small spectral-radius gap and positive correlation with the latent one-step Jacobian proxy: local stretching and contraction are being preserved.
- Good one-step error with poor spectral alignment: expect rollout drift or attractor distortion.

## Separability

- Low off-diagonal mutual information between `delta z_i` and `z_j` for `i != j`: the coordinate is closer to weakly coupled dynamics.
- High off-diagonal mutual information: the coordinate remains entangled and may not match Koopman-like modes.

## Follow-up choices

- Delay wins closure: increase memory or test explicit closure terms.
- Fast-slow wins separability but not spectrum: retune timescales or learn slow coordinates instead of fixing them by hand.
- Factor coordinates win closure and separability: promote those factors into targeted mining or structured residual models.
