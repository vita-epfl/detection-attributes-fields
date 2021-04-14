from random import randrange

import torch


class GradientForkNormalization(torch.autograd.Function):
    """Autograd function for MTL gradient fork-normalization layer."""

    @staticmethod
    def forward(ctx, input_, normalization, duplicates):
        ctx.normalization = normalization
        ctx.save_for_backward(input_)
        output = tuple(input_.clone() for _ in range(duplicates))
        return output


    @staticmethod
    def backward(ctx, *grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            input, = ctx.saved_tensors
            grad_input = torch.zeros_like(input)
            for n in range(grad_input.shape[0]):
                valid_gradout = [gradout[n] for gradout in grad_output if (
                    (gradout is not None)
                    and (torch.norm(gradout[n].view(-1), p=2).item() > 1e-8)
                )]
                if len(valid_gradout) == 0:
                    continue
                elif ctx.normalization == 'accumulation':
                    grad_input[n] = sum(valid_gradout)
                elif ctx.normalization == 'average':
                    grad_input[n] = sum(valid_gradout) / len(valid_gradout)
                elif ctx.normalization == 'power':
                    grad_input[n] = sum(valid_gradout) / (len(valid_gradout)**.5)
                elif ctx.normalization == 'sample':
                    grad_input[n] = valid_gradout[randrange(len(valid_gradout))]
                elif ctx.normalization == 'random':
                    weights = torch.distributions.dirichlet.Dirichlet(
                        torch.ones(len(valid_gradout))).sample()
                    grad_input[n] = sum([g*w.item()
                        for g, w in zip(valid_gradout, weights)])
        return grad_input, None, None


class MtlGradForkNorm(torch.nn.Module):
    """Multi-Task Learning Gradient Fork-Normalization layer.
    Normalize gradients joining at a fork during backward (forward pass left
        unchanged).

    Args:
        normalization (str): Type of normalization ('accumulation', 'average',
            'power', 'sample', 'random').
        duplicates (int): Max number of branches to normalize for.
    """

    def __init__(self,
                 normalization: str = 'accumulation',
                 duplicates: int = 1):
        super().__init__()
        if normalization not in ('accumulation', 'average', 'power',
                                 'sample', 'random'):
            raise ValueError(
                'unsupported normalization {}'.format(normalization))
        self.normalization = normalization
        self.duplicates = duplicates


    def extra_repr(self):
        return 'normalization={}, duplicates={}'.format(
            self.normalization, self.duplicates)


    def forward(self, input_):
        return GradientForkNormalization.apply(
            input_, self.normalization, self.duplicates)
