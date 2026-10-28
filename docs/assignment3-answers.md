# Assignment 3 — worked theory answers

## 1. Transposed convolution output size

For each spatial dimension, with dilation D = 1:

O = (I − 1)S − 2P + (K − 1) + OP + 1
  = (I − 1)S − 2P + K + OP.

Substituting I = 8, K = 4, S = 2, P = 1, OP = 1:

O = (8 − 1)·2 − 2·1 + 4 + 1 = 14 − 2 + 4 + 1 = 17.

Both dimensions are identical, so the output is **17 × 17**.

## 2. Increasing stride from 2 to 3

Holding the Question 1 values fixed:

O_new = (8 − 1)·3 − 2·1 + 4 + 1 = 21 − 2 + 4 + 1 = 24.

The output changes from **17 × 17 to 24 × 24**. More generally,

ΔO = (I − 1)(S_new − S_old) = (I − 1)(3 − 2) = I − 1.

Thus each spatial dimension increases by 7 for I = 8. The output size is not simply multiplied by 3/2 because the padding and kernel terms stay fixed.

## 3. General formula

For dilation 1, apply this formula independently to height and width:

**O = (I − 1)S − 2P + K + OP.**

If dilation D is included, the general formula is:

O = (I − 1)S − 2P + D(K − 1) + OP + 1.

Use the corresponding height or width values if the parameters are pairs.

## 4. Upsampling 16 × 16 to 32 × 32 without padding

Choose **kernel size 2 × 2, stride 2, padding 0, output padding 0**, with dilation 1.

O = (16 − 1)·2 − 2·0 + 2 + 0 = 30 + 2 = 32.

Therefore this configuration produces 32 × 32.

## 5. Batch normalization of [6, 8, 10, 6]

The mini-batch mean is:

μ = (6 + 8 + 10 + 6)/4 = 30/4 = 7.5.

Deviations from the mean are [−1.5, 0.5, 2.5, −1.5]. BatchNorm uses the population (biased) variance for normalization during training:

σ² = ((−1.5)² + 0.5² + 2.5² + (−1.5)²)/4
   = (2.25 + 0.25 + 6.25 + 2.25)/4
   = 11/4 = 2.75.

σ = √2.75 = √11/2 ≈ 1.658312.

With no affine transformation and ignoring epsilon for exact unit variance:

x̂ = (x − μ)/σ
  = [−3/√11, 1/√11, 5/√11, −3/√11]
  ≈ **[−0.904534, 0.301511, 1.507557, −0.904534]**.

Check: the sum is (−3 + 1 + 5 − 3)/√11 = 0, so the mean is 0.
The variance is (9 + 1 + 25 + 9)/(11·4) = 44/44 = 1.

In a numerical implementation, x̂ = (x − μ)/√(σ² + ε). With PyTorch's default ε = 10⁻⁵, the outputs are approximately [−0.904532, 0.301511, 1.507554, −0.904532], and variance is slightly less than 1. PyTorch uses an unbiased variance estimate for its running-variance update, which is separate from the variance used to normalize this training batch.

## 6. ReLU versus LeakyReLU

ReLU(x) = max(0, x) = x for x ≥ 0, and 0 for x < 0.

LeakyReLU(x) = x for x ≥ 0, and αx for x < 0, where α is a small positive constant (0.2 in this assignment).

Their key difference is the negative-input slope: 0 for ReLU, α for LeakyReLU. Away from zero, their derivatives are respectively 1 or 0 for ReLU, and 1 or α for LeakyReLU. Both have a kink at zero when α ≠ 1; implementations choose a convention for the derivative there.

## 7. Why prefer LeakyReLU?

For a negative preactivation, ReLU produces both a zero output and a zero local derivative. A neuron that receives only negative preactivations may therefore stop learning through that activation (the “dying ReLU” problem). LeakyReLU retains derivative α on the negative side, allowing a gradient to pass and the neuron's weights to continue changing. This can help optimization in deep networks, including GAN discriminators. It reduces this particular source of zero gradients but does not guarantee stable GAN training or prevent all vanishing gradients.
