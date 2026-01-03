import torch.nn as nn
import torch
import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from solvers.cg_utils import conjugate_gradient

class EquilibriumGrad(nn.Module):
    def __init__(self, linear_operator, nonlinear_operator, eta, minval = -1, maxval = 1):
        super(EquilibriumGrad,self).__init__()
        self.linear_op = linear_operator
        self.nonlinear_op = nonlinear_operator
        # self.eta = eta

        self.minval = minval
        self.maxval = maxval

        # Check if the linear operator has parameters that can be learned:
        # if so, register them to be learned as part of the network.
        linear_param_name = 'linear_param_'
        for ii, parameter in enumerate(self.linear_op.parameters()):
            parameter_name = linear_param_name + str(ii)
            self.register_parameter(name=parameter_name, param=parameter)

        self.register_parameter(name='eta', param=torch.nn.Parameter(torch.tensor(eta), requires_grad=True))


    def _linear_op(self, x):
        return self.linear_op.forward(x)

    def _linear_adjoint(self, x):
        return self.linear_op.adjoint(x)

    def set_initial_point(self, y):
        self.initial_point = self._linear_adjoint(y)

    def get_gradient(self, z, y):
        return self.linear_op.gramian(z)  - self._linear_adjoint(y) - self.nonlinear_op(z)

    def forward(self, z, y):
        z_tplus1 = z - self.eta * self.get_gradient(z, y)
        z_tplus1 = torch.clamp(z_tplus1, self.minval, self.maxval)
        return z_tplus1

class EquilibriumProxGradMRI(nn.Module):
    def __init__(self, linear_operator, nonlinear_operator, eta, minval = -1, maxval = 1):
        super(EquilibriumProxGradMRI,self).__init__()
        self.linear_op = linear_operator
        self.nonlinear_op = nonlinear_operator

        self.minval = minval
        self.maxval = maxval
        self.eta = eta

        # Check if the linear operator has parameters that can be learned:
        # if so, register them to be learned as part of the network.
        linear_param_name = 'linear_param_'
        for ii, parameter in enumerate(self.linear_op.parameters()):
            parameter_name = linear_param_name + str(ii)
            self.register_parameter(name=parameter_name, param=parameter)

    def _linear_op(self, x):
        return self.linear_op.forward(x)

    def _linear_adjoint(self, x):
        return self.linear_op.adjoint(x)

    def get_gradient(self, z, y):
        return self.linear_op.gramian(z)  - self._linear_adjoint(y)

    def forward(self, z, y):
        gradstep = z - self.eta * self.get_gradient(z, y)
        z_tplus1 = gradstep + self.nonlinear_op(gradstep)
        z_tplus1 = torch.clamp(z_tplus1, self.minval, self.maxval)
        return z_tplus1