import torch
import torch.nn as nn
import numpy as np

try:
    import gudhi as gd
    from gudhi.wasserstein import wasserstein_distance
except ImportError:
    print("="*50)
    print("ERROR: Missing required GUDHI library for 3D topological loss.")
    print("Please install it by running: pip install gudhi")
    print("="*50)
    raise

class TopologicalLoss3D(nn.Module):
    """
    Calculates a 3D topological loss by adapting the logic from topo_losses.py.
    This loss operates in pixel-space, providing direct gradients to critical pixels.

    It combines two main ideas:
    1.  Signal-Noise Decomposition: Separates topological features into important "signals"
        and unimportant "noise" based on a persistence threshold.
    2.  Differentiated Loss:
        -   Signal Consistency: Forces the student's signal features to match the teacher's.
        -   Noise Removal: Forces the student's noise features to be eliminated.
    """
    def __init__(self, homology_dimensions=(0, 1), pd_threshold=0.1):
        """
        Args:
            homology_dimensions (tuple): Homology dimensions to consider (e.g., (0, 1) for components and loops).
            pd_threshold (float): Persistence threshold to separate signal from noise.
        """
        super().__init__()
        self.homology_dimensions = homology_dimensions
        self.pd_threshold = pd_threshold

    def _compute_pd_and_coords(self, volume: np.ndarray):
        """
        Computes persistence diagrams and the coordinates of their critical pixels,
        aligned with the GUDHI v3.11.0 API.
        """
        if not isinstance(volume, np.ndarray):
            volume = volume.detach().cpu().numpy()

        # Invert the volume because GUDHI's CubicalComplex finds sublevel set homology.
        inverted_volume = 1.0 - volume
        
        # Ensure the input is Fortran-contiguous, as GUDHI expects.
        # This is a critical step to prevent memory layout mismatches.
        inverted_volume = np.asfortranarray(inverted_volume)

        cubical_complex = gd.CubicalComplex(top_dimensional_cells=inverted_volume)
        cubical_complex.compute_persistence()

        # Get cofaces (indices of top-dimensional cells) for persistence pairs.
        # This is the correct function for GUDHI v3.11.0.
        regular_pairs_by_dim, _ = cubical_complex.cofaces_of_persistence_pairs()

        all_diagrams = []
        all_coords = []
        
        # Get a flat array of all filtration values for easy lookup.
        # We must explicitly flatten it in Fortran order to match the indices from GUDHI.
        filtration_values_flat = cubical_complex.top_dimensional_cells().flatten(order='F')
        original_shape = inverted_volume.shape

        for dim_to_process in self.homology_dimensions:
            diagram = []
            coords = []

            # Check if there are pairs for the current dimension
            if dim_to_process < len(regular_pairs_by_dim):
                dim_pairs = regular_pairs_by_dim[dim_to_process]
                
                for birth_cell_idx, death_cell_idx in dim_pairs:
                    # The indices point to the flattened array of top-dimensional cells.
                    birth_val_inverted = filtration_values_flat[birth_cell_idx]
                    death_val_inverted = filtration_values_flat[death_cell_idx]

                    # Undo the inversion to get original persistence values
                    birth_val = 1.0 - death_val_inverted
                    death_val = 1.0 - birth_val_inverted
                    
                    # Ensure birth < death
                    if birth_val >= death_val:
                        continue
                        
                    diagram.append([birth_val, death_val])

                    # Convert flat indices to 3D coordinates using Fortran order.
                    birth_coords = np.array(np.unravel_index(birth_cell_idx, original_shape, order='F'))
                    death_coords = np.array(np.unravel_index(death_cell_idx, original_shape, order='F'))
                    coords.append([birth_coords, death_coords])

            all_diagrams.append(np.array(diagram) if diagram else np.empty((0, 2)))
            all_coords.append(np.array(coords) if coords else np.empty((0, 2, 3)))

        return all_diagrams, all_coords

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Calculates the 3D topological loss.
        """
        if y_pred.dim() != 3 or y_true.dim() != 3:
            raise ValueError("Input tensors must be 3D.")

        pred_np = y_pred.detach().cpu().numpy()
        true_np = y_true.detach().cpu().numpy().astype(float)

        if pred_np.max() - pred_np.min() < 0.01 or true_np.sum() == 0:
            return torch.tensor(0.0, device=y_pred.device)

        pd_pred_list, coords_pred_list = self._compute_pd_and_coords(pred_np)
        pd_true_list, coords_true_list = self._compute_pd_and_coords(true_np)

        # Initialize weight and reference maps, similar to topo_losses.py
        weight_map = torch.zeros_like(y_pred)
        ref_map = torch.zeros_like(y_pred)
        
        total_loss = 0.0
        
        # Process each homology dimension separately
        for i in range(len(self.homology_dimensions)):
            pd_pred, coords_pred = pd_pred_list[i], coords_pred_list[i]
            pd_true, coords_true = pd_true_list[i], coords_true_list[i]
            
            if pd_pred.shape[0] == 0: continue

            # Signal-Noise Decomposition
            persistence_pred = pd_pred[:, 1] - pd_pred[:, 0]
            signal_idx_pred = np.where(persistence_pred >= self.pd_threshold)[0]
            noise_idx_pred = np.where(persistence_pred < self.pd_threshold)[0]

            persistence_true = pd_true[:, 1] - pd_true[:, 0]
            signal_idx_true = np.where(persistence_true >= self.pd_threshold)[0]

            pd_pred_signal = pd_pred[signal_idx_pred]
            pd_true_signal = pd_true[signal_idx_true]

            # Find optimal matching between signal diagrams
            cost, matchings = wasserstein_distance(pd_pred_signal, pd_true_signal, matching=True, order=2)
            
            # Identify matched and unmatched student signal points
            off_diagonal_matches = matchings[matchings[:, 1] != -1]
            unmatched_pred_indices_in_signal = matchings[matchings[:, 1] == -1, 0]

            # --- 1. Consistency Loss for Matched Signal ---
            for pred_match_idx, true_match_idx in off_diagonal_matches:
                original_pred_idx = signal_idx_pred[pred_match_idx]
                original_true_idx = signal_idx_true[true_match_idx]

                b_coord_pred = tuple(coords_pred[original_pred_idx, 0].astype(int))
                d_coord_pred = tuple(coords_pred[original_pred_idx, 1].astype(int))
                
                # Target for birth pixel is birth value of teacher
                weight_map[b_coord_pred] = 1
                ref_map[b_coord_pred] = pd_true[original_true_idx, 0]

                # Target for death pixel is death value of teacher
                weight_map[d_coord_pred] = 1
                ref_map[d_coord_pred] = pd_true[original_true_idx, 1]

            # --- 2. Noise Removal Loss (for student's noise AND unmatched signal) ---
            # Combine noisy points and unmatched signal points into one list of points to remove
            unmatched_signal_original_indices = signal_idx_pred[unmatched_pred_indices_in_signal]
            indices_to_remove = np.concatenate([noise_idx_pred, unmatched_signal_original_indices])

            for idx in indices_to_remove:
                b_coord = tuple(coords_pred[idx, 0].astype(int))
                d_coord = tuple(coords_pred[idx, 1].astype(int))
                
                # The goal is to make birth_value = death_value
                # Force birth pixel to take the value of the death pixel
                weight_map[b_coord] = 1
                ref_map[b_coord] = y_pred[d_coord]

                # Force death pixel to take the value of the birth pixel
                weight_map[d_coord] = 1
                ref_map[d_coord] = y_pred[b_coord]

        # Final loss calculation (weighted MSE in pixel-space)
        loss = (((y_pred * weight_map) - (ref_map * weight_map)) ** 2).sum()
        
        return loss 