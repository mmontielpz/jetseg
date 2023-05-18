import torch
import torch.nn as nn

DEBUG = False


class REUFunction(torch.autograd.Function):
    """The Rectified Exponential Unit (REU) activation function.

        REU(x) = max(0, x) + min(0, x*e^(x)

        Function
        if x > 0 --> x
        if x <= 0 --> x * e^(x)

        Gradient
        if x >= 0 --> 1
        if x < 0 --> e^(x) * (x + 1)

    """

    @staticmethod
    def forward(ctx, data: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass."""

        ctx.save_for_backward(data)
        return torch.where(data <= 0.0, data * torch.exp(data), data)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """Performs a backpropagation."""

        (data, ) = ctx.saved_tensors
        data = data.double()
        grad = torch.where(data >= 0.0, 1.0, torch.exp(data) * (data + 1))
        return grad_output * grad


class REU(nn.Module):

    def forward(self, x):
        return REUFunction.apply(x)


class TanhExpFunction(torch.autograd.Function):
    """ The TANH Exponential Activation Function (TanhExp)

        #Todo: Missing general expression :(
        TanhExp(x) = max(0, x) + min(0, x*e^(x)

        Function (Forward)
        if x >= 0 --> x * tanh(e^x)
        otherwise --> 0

        Gradient
        if x
        if x <= -3 --> e^(x) * (x + 1)
        if x >= 3 --> 1
        otherwise --> (2x + 3)/6
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass."""

        # Save the input tensor for use in the backward pass
        ctx.save_for_backward(x)
        return x * torch.tanh(torch.exp(x))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """Performs a backpropagation."""

        x, = ctx.saved_tensors
        x = x.double()

        # Compute the gradient of the function with respect to x
        grad_x = grad_output * (torch.tanh(torch.exp(x)) + x * (1 - torch.tanh(torch.exp(x))**2) * torch.exp(x))

        return grad_x


class TanhExp(nn.Module):

    def forward(self, x):
        return TanhExpFunction.apply(x)


# Test Activation function
if DEBUG:

    torch.manual_seed(0)

    sqreu = TanhExp()

    data = torch.rand(4, dtype=torch.double, requires_grad=True)

    print(sqreu(data))

    if torch.autograd.gradcheck(sqreu, data, eps=1e-8, atol=1e-7):
        print("gradcheck successful")
    else:
        print("gradcheck unsuccessful")
