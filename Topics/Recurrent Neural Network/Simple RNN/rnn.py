import torch
from torch import nn


class RNN(nn.Module):
    """
    Implementation of the
    [Elman network](https://en.wikipedia.org/wiki/Recurrent_neural_network#Elman_networks_and_Jordan_networks).
    This is also the implementation used in
    [`torch.nn.RNN`](https://en.wikipedia.org/wiki/Recurrent_neural_network#Elman_networks_and_Jordan_networks)

    This network should receive an input of shape `(batch_size, sequence_len, in_features)`
    and will output an output tensor of shape `(batch_size, sequence_len, hidden_size)`
    as well as the last hidden state of shape `(batch_size, hidden_size)`.

    One can also pass in an optional hidden state of shape `(batch_size, hidden_size)`.
    """
    def __init__(self,
        in_features: int,
        hidden_size: int,
        activation_fn: nn.Module | None = None
    ):
        """
        Initalize a RNN.

        :param int in_features: The dimension of input features. The input
            tensors should have shape `(batch_size, sequence_len, in_features)`.
        :param int hidden_size: The dimension of the hidden state. This is also the
            dimension of the output.
        :param None | torch.nn.Module activation_fn: The activation function to apply
            on the hidden state of the network.
        """
        super().__init__()
        if activation_fn is None: activation_fn = nn.ReLU()
        self.act_fn = activation_fn


        self.w_in_hidden     = nn.Parameter(torch.Tensor(hidden_size, in_features))
        self.w_hidden_hidden = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_in         = nn.Parameter(torch.Tensor(hidden_size))
        self.bias_hidden     = nn.Parameter(torch.Tensor(hidden_size))

        sqrt_k = torch.sqrt(torch.tensor(1 / hidden_size))
        self.w_in_hidden.data.uniform_(-sqrt_k, sqrt_k)
        self.w_hidden_hidden.data.uniform_(-sqrt_k, sqrt_k)
        self.bias_in.data.uniform_(-sqrt_k, sqrt_k)
        self.bias_hidden.data.uniform_(-sqrt_k, sqrt_k)

    def forward(self,
        sequence: torch.Tensor,
        hidden_state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor]:
        if hidden_state is None:
            hidden_state = torch.zeros((sequence.shape[0], self.bias_hidden.shape[0]))
            
        output = torch.stack([
            (hidden_state := self.act_fn(
                sequence[:, i_seq] @ self.w_in_hidden.T + self.bias_in +
                hidden_state @ self.w_hidden_hidden.T + self.bias_hidden
            )) for i_seq in range(sequence.shape[1])
        ], dim=1)
        return output, hidden_state
    
    def __repr__(self) -> str:
        return (
            "RNN(" +
            f"in_features={self.w_in_hidden.shape[1]}, " +
            f"hidden_size={self.w_in_hidden.shape[0]}, " +
            f"activation_fn={self.act_fn})"
        )