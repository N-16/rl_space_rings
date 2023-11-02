import torch
from torch.nn import Parameter

class WrapperNet(torch.nn.Module):
    def __init__(
            self,
            qnet,
            discrete_output_sizes,
            dueling_formward
    ):
        """
        Wraps the VisualQNetwork adding extra constants and dummy mask inputs
        required by runtime inference with Sentis.

        For environment continuous actions outputs would need to add them
        similarly to how discrete action outputs work, both in the wrapper
        and in the ONNX output_names / dynamic_axes.
        """
        super(WrapperNet, self).__init__()
        self.qnet = qnet

        # version_number
        #   MLAgents1_0 = 2   (not covered by this example)
        #   MLAgents2_0 = 3
        version_number = torch.Tensor([3])
        self.version_number = Parameter(version_number, requires_grad=False)

        # memory_size
        # TODO: document case where memory is not zero.
        memory_size = torch.Tensor([0])
        self.memory_size = Parameter(memory_size, requires_grad=False)

        # discrete_action_output_shape
        output_shape=torch.Tensor([discrete_output_sizes])
        self.discrete_shape = Parameter(output_shape, requires_grad=False)

        self.dueling_forward = dueling_formward


    # if you have discrete actions ML-agents expects corresponding a mask
    # tensor with the same shape to exist as input
    def forward(self, visual_obs: torch.tensor, mask: torch.tensor):
        if self.dueling_forward:
            _, qnet_result = self.qnet(visual_obs)
        else:
            qnet_result = self.qnet(visual_obs)
        # Connect mask to keep it from getting pruned
        # Mask values will be 1 if you never call SetActionMask() in
        # WriteDiscreteActionMask()
        qnet_result = torch.mul(qnet_result, mask)
        action = torch.argmax(qnet_result, dim=1, keepdim=True)
        return [action], self.discrete_shape, self.version_number, self.memory_size



    