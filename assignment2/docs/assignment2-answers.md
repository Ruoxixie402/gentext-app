# Assignment 2: worked CNN arithmetic

For a convolution or pooling layer, calculate each spatial dimension using

\[
O = \left\lfloor\frac{I + 2P - D(K-1) - 1}{S}\right\rfloor + 1.
\]

Here \(I\) is input size, \(K\) is kernel size, \(P\) is padding on each side,
\(S\) is stride, and \(D\) is dilation. All these questions use \(D=1\), so

\[
O = \left\lfloor\frac{I + 2P - K}{S}\right\rfloor + 1.
\]

Shapes below use height × width × channels; PyTorch uses batch × channels × height × width.

## Question 1

Input: 32 × 32 × 3. There are 8 filters, each covering all 3 input channels.
Kernel size \(K=5\), stride \(S=1\), padding \(P=0\).

\[
H_{out}=W_{out}=\left\lfloor\frac{32+0-5}{1}\right\rfloor+1=27+1=28.
\]

Each filter produces one output channel, so the output is **28 × 28 × 8**.

## Question 2

For a 5 × 5 kernel at stride 1, “same” padding requires
\(P=(K-1)/2=(5-1)/2=2\) on each side.

\[
H_{out}=W_{out}=\left\lfloor\frac{32+2(2)-5}{1}\right\rfloor+1=31+1=32.
\]

The spatial size stays 32 × 32; 8 filters still give 8 channels.
The output is **32 × 32 × 8**.

## Question 3

Input size \(I=64\), kernel \(K=3\), stride \(S=2\), padding \(P=0\).

\[
H_{out}=W_{out}=\left\lfloor\frac{64-3}{2}\right\rfloor+1
=\lfloor30.5\rfloor+1=30+1=31.
\]

The output spatial size is **31 × 31**. The number of output channels
depends on the number of filters, which the question does not specify.
The floor excludes a final incomplete kernel placement.

## Question 4

Input size \(I=16\), pooling kernel \(K=2\), stride \(S=2\), padding \(P=0\).

\[
H_{out}=W_{out}=\left\lfloor\frac{16-2}{2}\right\rfloor+1=7+1=8.
\]

The output is **8 × 8** for one feature map. Pooling preserves channels,
so an input 16 × 16 × C would produce 8 × 8 × C.

## Question 5

Each 3 × 3, stride-1 convolution uses “same” padding: \(P=(3-1)/2=1\).
For the first layer:

\[
O_1=\left\lfloor\frac{128+2(1)-3}{1}\right\rfloor+1=127+1=128.
\]

For the second layer:

\[
O_2=\left\lfloor\frac{128+2(1)-3}{1}\right\rfloor+1=128.
\]

The output spatial shape is **128 × 128**. If the second layer has F filters,
the full output is 128 × 128 × F. The question does not specify F.

## Question 6

`model.train()` sets the module and its children to training mode. It does
not itself run optimization or enable gradient calculation.

A newly constructed PyTorch model is already in training mode. If it has
never been put into evaluation mode, removing this call does not change its
behavior. If `model.eval()` was called previously (for example, during the
previous epoch's validation), omitting `model.train()` leaves it in evaluation mode.

This matters for mode-dependent layers. Dropout stops randomly dropping
activations in evaluation mode. Batch normalization uses stored running
statistics rather than the current batch and stops updating those statistics.
Gradients can still be calculated and an optimizer can still update parameters
in evaluation mode; `eval()` is separate from `no_grad()` and `inference_mode()`.

The exact architecture in this assignment contains only convolution, ReLU,
max pooling, flattening, and linear layers. These layers have the same behavior
in both modes, so removing `train()` does not change this particular network's
outputs or learning, assuming gradient calculation and optimization are unchanged.
Keeping the call makes the training loop correct if mode-dependent layers are
added later.

## Architecture verification (Part 1)

| Layer | Output shape (excluding batch) | Trainable parameters |
| --- | --- | --- |
| Input | 3 × 64 × 64 | 0 |
| Conv 3→16, 3×3, stride 1, padding 1 | 16 × 64 × 64 | 16(3·3·3 + 1) = 448 |
| ReLU | 16 × 64 × 64 | 0 |
| Max pool 2×2, stride 2 | 16 × 32 × 32 | 0 |
| Conv 16→32, 3×3, stride 1, padding 1 | 32 × 32 × 32 | 32(16·3·3 + 1) = 4,640 |
| ReLU | 32 × 32 × 32 | 0 |
| Max pool 2×2, stride 2 | 32 × 16 × 16 | 0 |
| Flatten | 8,192 | 0 |
| Linear 8,192→100 | 100 | 8,192·100 + 100 = 819,300 |
| ReLU | 100 | 0 |
| Linear 100→10 | 10 | 100·10 + 10 = 1,010 |

Total parameters: 448 + 4,640 + 819,300 + 1,010 = **825,398**.
The final layer produces logits. Cross-entropy loss consumes logits during
training; softmax converts them to class probabilities during inference.
