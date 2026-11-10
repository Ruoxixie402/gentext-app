"""Executable verification of Assignment 4 Questions 6–9."""
import torch


def main():
    x = torch.tensor([2.0], requires_grad=True)
    (x ** 2 + 3 * x).backward()
    print('Q6(a): x.grad =', x.grad)
    x = torch.tensor([2.0], requires_grad=False)
    try:
        (x ** 2 + 3 * x).backward()
    except RuntimeError:
        print('Q6(b): backward raises RuntimeError; x.grad =', x.grad)
    print('Q6(c)/Q7(c): default requires_grad =', torch.tensor([2.0]).requires_grad)
    x = torch.tensor([2.0], requires_grad=True)
    w = torch.tensor([1.0, 3.0])
    (w[0] * x ** 2 + w[1] * x).backward()
    print('Q7(a): x.grad =', x.grad, '; w.grad =', w.grad)
    x = torch.tensor([2.0], requires_grad=True)
    w = torch.tensor([1.0, 3.0], requires_grad=True)
    (w[0] * x ** 2 + w[1] * x).backward()
    print('Q7(b): x.grad =', x.grad, '; w.grad =', w.grad)
    x = torch.tensor([1.0], requires_grad=True)
    y = x * 3
    z = y.detach()
    try:
        (z * 2).backward()
    except RuntimeError:
        print('Q8 original: backward raises RuntimeError')
    z = y.clone()
    (z * 2).backward()
    print('Q8 corrected: x.grad =', x.grad)
    x = torch.tensor([1.0], requires_grad=True)
    (x * 2).backward()
    print('Q9 first backward: x.grad =', x.grad)
    (x * 3).backward()
    print('Q9 accumulated: x.grad =', x.grad)
    x.grad = None
    (x * 3).backward()
    print('Q9 after clearing: x.grad =', x.grad)


if __name__ == '__main__':
    main()
