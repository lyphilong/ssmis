"""
Utility functions for Topological Progression/Evolution Loss.

This module requires the installation of GUDHI, Persim, and scikit-learn:
    pip install gudhi persim scikit-learn

Main components:
1.  Compute persistence diagrams (PD) for 2D slices.
2.  Vectorize PDs into Persistence Images (PI).
3.  Calculate the progression loss based on the smoothness of the PI sequence.
"""

import torch
import torch.nn as nn
import numpy as np

# Try to import specialized libraries and provide helpful error messages.
try:
    import gudhi as gd
    from persim import PersistenceImager
except ImportError:
    print("="*50)
    print("ERROR: Missing required libraries for topological progression loss.")
    print("Please install them by running: pip install gudhi persim scikit-learn")
    print("="*50)
    # Re-raise the error to stop execution if the libraries are essential
    raise

class TopologicalProgressionLoss(nn.Module):
    """
    Calculates the Topological Progression Loss for a 3D volume.

    This loss encourages the topological features (represented by Persistence Images)
    to evolve smoothly across the slices of a 3D volume.
    """
    def __init__(self, pi_resolution=(20, 20), pi_sigma=0.01):
        """
        Args:
            pi_resolution (tuple): The resolution (height, width) of the Persistence Image.
            pi_sigma (float): The variance of the Gaussian kernel for the PI.
        """
        super().__init__()
        
        # Calculate pixel_size from the desired resolution.
        # We assume the same resolution for both axes for simplicity, as required by some persim versions.
        pixel_size = 1.0 / pi_resolution[0]

        # Initialize PersistenceImager.
        # This version is made compatible with older 'persim' libraries that expect
        # 'pixel_size' as a float and do not have a 'resolution' parameter.
        self.imager = PersistenceImager(
            pixel_size=pixel_size,
            kernel_params={'sigma': [[pi_sigma, 0.0], [0.0, pi_sigma]]}
        )
             
    def _compute_pd_for_slice(self, slice_2d: np.ndarray):
        """Computes the persistence diagram for a single 2D slice using sublevel set filtration."""
        if torch.is_tensor(slice_2d):
            slice_2d = slice_2d.detach().cpu().numpy()

        # Create a GUDHI CubicalComplex from the 2D slice
        cubical_complex = gd.CubicalComplex(top_dimensional_cells=slice_2d)
        cubical_complex.compute_persistence()
        
        # We are interested in 0-dimensional homology (connected components)
        pd_h0 = cubical_complex.persistence_intervals_in_dimension(0)
        
        # Guard against GUDHI returning a 1D array for a single persistence pair.
        # If it's 1D (e.g., shape (2,) instead of (1, 2)), reshape it to be 2D.
        if pd_h0.ndim == 1 and pd_h0.size > 0:
            pd_h0 = pd_h0.reshape(1, -1)
            
        # Filter out infinite persistence pairs (the main background component)
        # This check is now safe because pd_h0 is guaranteed to be 2D.
        pd_h0_finite = pd_h0[pd_h0[:, 1] != np.inf]
        
        return pd_h0_finite

    def forward(self, volume_3d: torch.Tensor):
        """
        Calculates the progression loss for a single 3D probability volume.

        Args:
            volume_3d (torch.Tensor): A 3D tensor of shape (H, W, D).

        Returns:
            torch.Tensor: The scalar topological progression loss.
        """
        if volume_3d.dim() != 3:
            raise ValueError("Input volume must be a 3D tensor.")

        num_slices = volume_3d.shape[-1]
        if num_slices < 2:
            return torch.tensor(0.0, device=volume_3d.device)

        # 1. Compute Persistence Diagram for each slice
        diagrams = []
        for d_idx in range(num_slices):
            slice_2d = volume_3d[..., d_idx]
            if slice_2d.max() - slice_2d.min() > 0.1:
                diagrams.append(self._compute_pd_for_slice(slice_2d))
            else:
                diagrams.append(np.array([]))

        # 2. Vectorize PDs into Persistence Images (PIs)
        all_diagrams_flat = [d for d in diagrams if d.shape[0] > 0]
        if not all_diagrams_flat:
             return torch.tensor(0.0, device=volume_3d.device)
        
        self.imager.fit(all_diagrams_flat, skew=True)
        persistence_images = self.imager.transform(diagrams, skew=True)
        
        pi_tensor = torch.from_numpy(np.array(persistence_images)).float().to(volume_3d.device)
        pi_vectors = pi_tensor.view(num_slices, -1)
        
        # 3. Calculate the progression loss (sum of squared L2 distances)
        diffs = pi_vectors[:-1] - pi_vectors[1:]
        progression_loss = torch.sum(diffs**2)

        return progression_loss / (num_slices - 1) 