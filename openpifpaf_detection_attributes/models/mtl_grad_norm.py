from random import randrange

import numpy as np
import torch


class GradientNormalization(torch.autograd.Function):
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
                    weights = np.random.dirichlet(np.ones(len(valid_gradout)))
                    grad_input[n] = sum([g*w
                        for g, w in zip(valid_gradout, weights)])
        return grad_input, None, None


class MtlGradNorm(torch.nn.Module):
    def __init__(self, normalization='accumulation', duplicates=1):
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
        return GradientNormalization.apply(
            input_, self.normalization, self.duplicates)
