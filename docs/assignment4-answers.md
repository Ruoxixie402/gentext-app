# Assignment 4 — worked answers

## Question 1 — sinusoidal timestep embedding

Let the embedding dimension d be even, M = 10000 be the maximum period parameter,
and j = 0,...,d/2−1. This implementation follows the interleaved transformer convention:

ω_j = exp(−(2j/d) ln M) = M^(−2j/d).

For zero-based embedding coordinate i:

- e_(2j)(t) = sin(t ω_j).
- e_(2j+1)(t) = cos(t ω_j).

Equivalently, e_i(t) is sin(t/M^(i/d)) for even i, and
cos(t/M^((i−1)/d)) for odd i. Each adjacent sine/cosine pair shares a frequency.
The vector is then processed by a learned MLP and injected into UNet blocks.
The maximum-period parameter controls the frequency scale; t need not be bounded by M.

Conventions vary: some diffusion implementations concatenate all cosine coordinates
then all sine coordinates, and some space frequencies using d/2−1 rather than d/2.
Coordinate ordering and spacing must be stated; the formulas and numbers here match
`app/cifar10/models.py` exactly. If the class implementation uses another convention,
its vector ordering or frequency spacing will differ.

## Question 2 — d = 8, t = 1

With the convention above, j = 0,1,2,3, and the exponents 2j/8 are
0, 1/4, 1/2, 3/4. Therefore:

ω = [10000^0, 10000^(−1/4), 10000^(−1/2), 10000^(−3/4)]
  = [1, 0.1, 0.01, 0.001].

Multiply by t = 1, then apply sine/cosine in radians:

e(1) = [sin(1), cos(1), sin(0.1), cos(0.1),
        sin(0.01), cos(0.01), sin(0.001), cos(0.001)]

≈ [0.84147098, 0.54030231, 0.09983342, 0.99500417,
   0.00999983, 0.99995000, 0.00100000, 0.99999950].

## Question 3 — transformer positional encoding versus diffusion time embedding

Both encode a scalar index using sine and cosine functions at multiple frequencies.
This provides a fixed continuous representation that the network can use to distinguish
indices. In a transformer, the index describes token position, conveying sequence order;
its encoding is commonly added to token embeddings. In diffusion, t identifies the
noise level or diffusion step. Its embedding conditions the denoiser, commonly through
an MLP and additions or scale/shift operations in multiple UNet blocks. It does not
represent a pixel's spatial location or a token's place in a sentence.

## Question 4 — three stride-2 downsampling blocks

Assuming each block halves both spatial dimensions using appropriate kernel/padding:

64 × 64 -> 32 × 32 -> 16 × 16 -> 8 × 8.

Each dimension is 64/2³ = 8, so the bottleneck is **8 × 8**.
Stride alone does not specify the exact size without kernel, padding, and dilation.
For example, K=4, S=2, P=1, D=1 gives floor((I+2P−D(K−1)−1)/S+1)=I/2
for the even input sizes here. The CIFAR implementation uses two such blocks because
its images are 32×32; this question separately specifies a 64×64 image and three blocks.

## Question 5 — UNet output and diffusion loss

For the epsilon-prediction DDPM used here, the UNet receives x_t and t and outputs
an estimate ε_θ(x_t,t) of the added Gaussian noise. The output has the same shape as
the image, e.g. (B,3,32,32) for CIFAR-10.

Define β_t as the variance schedule, α_t=1−β_t, and
ᾱ_t=∏_(s=1)^t α_s. During training:

1. Draw a clean image x_0, timestep t, and ε ~ N(0,I).
2. Construct x_t = sqrt(ᾱ_t)x_0 + sqrt(1−ᾱ_t)ε.
3. Predict ε̂ = ε_θ(x_t,t).
4. Minimize L_simple = E[||ε−ε_θ(x_t,t)||²], implemented as the mean
   squared error over batch, channels, and pixels.

The known sampled ε is the target. Backpropagation updates θ, not the training images
or the target noise. During generation, predicted noise estimates help recover an
estimate of x_0 and parameterize each reverse denoising step. Other diffusion
parameterizations can predict x_0 or velocity v instead; the target and equations
must match the chosen parameterization.

## Question 6 — basic gradients

### (a)

y=x²+3x, so dy/dx=2x+3. At x=2:

dy/dx=2(2)+3=7.

The supplied code prints `x.grad = tensor([7.])`.

### (b)

With `requires_grad=False`, x does not request gradients. In this particular code,
there are no other gradient-tracked inputs, so y has no backward graph.
`y.backward()` raises RuntimeError (the tensor does not require grad and does not
have a grad_fn); x.grad remains None. Merely having some input with requires_grad=False
does not prevent a graph when another input, such as trainable weights, requires grad.

### (c)

`torch.tensor` defaults to `requires_grad=False`. Thus creating a tensor without the
flag does not request gradient tracking for that tensor. Operations involving other
gradient-requiring tensors may still be tracked. See the PyTorch documentation:
https://docs.pytorch.org/docs/stable/generated/torch.tensor.html

## Question 7 — introducing weights

### (a)

The supplied weights are created as `w=torch.tensor([1.0,3.0])`, so w.requires_grad
is False. The computation tracks gradients for x, but does not accumulate gradients
for w. Output:

```text
x.grad = tensor([7.])
w.grad = None
```

### (b)

For y=w_0 x²+w_1 x:

∂y/∂w_0=x²=2²=4,
∂y/∂w_1=x=2,
∂y/∂x=2w_0 x+w_1=2(1)(2)+3=7.

```python
import torch
x = torch.tensor([2.0], requires_grad=True)
w = torch.tensor([1.0, 3.0], requires_grad=True)
y = w[0] * x**2 + w[1] * x
y.backward()
print('x.grad =', x.grad)  # tensor([7.])
print('w.grad =', w.grad)  # tensor([4., 2.])
```

If only weight gradients are needed, x can have requires_grad=False while w remains True.
The weight gradient is still [4,2], and x.grad is None.

### (c)

The default is again requires_grad=False. Floating-point trainable weights must
explicitly request gradients (or be wrapped as nn.Parameter). Integer tensors cannot
require gradients. Enabling gradients does not itself update a tensor; an optimizer
step or an explicit no-grad update is still needed.

## Question 8 — breaking the graph

`y=x*3` depends on x, but `z=y.detach()` removes the graph connection. Therefore
`w=z*2` has no grad_fn and does not require gradients. Calling w.backward() raises
RuntimeError. Keep z connected by using y itself or a differentiable clone:

```python
import torch
x = torch.tensor([1.0], requires_grad=True)
y = x * 3
z = y.clone()  # Retains the differentiable connection to x.
w = z * 2
w.backward()
print(x.grad)  # tensor([6.])
```

By the chain rule, dw/dx = (dw/dz)(dz/dy)(dy/dx)=2·1·3=6.
`z=y.detach().requires_grad_(True)` would create a new leaf and permit gradients
with respect to z, but it would NOT reconnect z to x; x.grad would remain None.

## Question 9 — gradient accumulation

The first backward adds dy_1/dx=2, so x.grad is [2]. The second backward adds
dy_2/dx=3 to the existing buffer, giving [2+3]=[5]. PyTorch accumulates leaf gradients
instead of replacing them. These are two newly constructed graphs, so retain_graph
is not needed.

Clear the gradient between independent updates:

```python
import torch
x = torch.tensor([1.0], requires_grad=True)
y1 = x * 2
y1.backward()
print(x.grad)  # tensor([2.])
x.grad = None  # Or x.grad.zero_() when a gradient buffer exists.
y2 = x * 3
y2.backward()
print(x.grad)  # tensor([3.])
```

For model training, call `optimizer.zero_grad(set_to_none=True)` before a new
independent optimization step. Intentional gradient accumulation across microbatches
is valid when the loss scaling and update frequency are chosen accordingly.

## EBM implementation connection

The density is p_θ(x)=exp(−E_θ(x))/Z_θ. Negative log-likelihood has gradient
E_data[∇_θ E_θ(x)]−E_model[∇_θ E_θ(x)]. We approximate the model expectation
using finite-step replay-initialized Langevin chains and minimize
mean(E(real))−mean(E(negative)), plus a small squared-energy regularizer.

During sampling, model parameters are frozen and x.requires_grad=True.
`torch.autograd.grad(E(x).sum(), x)` returns only ∂E/∂x; image updates run under
no_grad, and samples are detached between steps. During model fitting, the weights
are trainable again and negative samples remain detached. Thus optimizer descent
lowers real-image energy and raises negative-image energy without differentiating
through the sampling trajectory. Finite-step chains with image clamping are a
contrastive approximation, not exact likelihood optimization or guaranteed mixing.
