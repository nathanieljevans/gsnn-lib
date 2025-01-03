import torch
import torch.nn as nn

class EnergyDistanceLoss(nn.Module):
    """
    Energy distance between two empirical distributions p_samples and q_samples.

    Shape:
        - p_samples: (n_samples_p, n_features)
        - q_samples: (n_samples_q, n_features)

    Returns:
        A scalar tensor representing the energy distance between the two distributions.
    """
    def __init__(self):
        super(EnergyDistanceLoss, self).__init__()

    def forward(self, p_samples: torch.Tensor, q_samples: torch.Tensor) -> torch.Tensor:
        """
        Computes the energy distance between two sets of samples:
            - p_samples: Samples from distribution P
            - q_samples: Samples from distribution Q

        energy_distance = 2 * E[||X - Y||] - E[||X - X'||] - E[||Y - Y'||]
        """
        # Pairwise distances within p_samples
        dist_x = torch.cdist(p_samples, p_samples, p=2)  # shape: (n_samples_p, n_samples_p)
        
        # Pairwise distances within q_samples
        dist_y = torch.cdist(q_samples, q_samples, p=2)  # shape: (n_samples_q, n_samples_q)
        
        # Pairwise distances between p_samples and q_samples
        dist_xy = torch.cdist(p_samples, q_samples, p=2) # shape: (n_samples_p, n_samples_q)

        # Compute the empirical means of the distances
        mean_dist_x = dist_x.mean()    # ~ E[||X - X'||]
        mean_dist_y = dist_y.mean()    # ~ E[||Y - Y'||]
        mean_dist_xy = dist_xy.mean()  # ~ E[||X - Y||]

        # Energy distance
        energy_distance = 2.0 * mean_dist_xy - mean_dist_x - mean_dist_y
        return energy_distance
