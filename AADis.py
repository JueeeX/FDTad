import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from collections import OrderedDict


class AdaptiveAttentionDistillation:
    """
    Implementation of Adaptive Attention Distillation (AADis) mechanism.
    This class manages the knowledge transfer between teacher and student models.
    """

    def __init__(self, teacher_model, student_model, device='cuda'):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.device = device

    def get_attention_matrices(self, model, x):
        """
        Extract attention matrices from the model.
        In a real implementation, you would modify the model to return
        attention matrices during forward pass.

        Args:
            model: The model to extract attention matrices from
            x: Input data

        Returns:
            Dictionary of attention matrices and value relationship matrices
        """
        # This is a simplified version for demonstration
        # In an actual implementation, you would hook into the model's SHSA modules
        # and extract Q, K, V matrices during forward pass

        # For this example, we'll assume:
        # - Each SHSA module stores its attention matrix in a list during forward pass
        # - We can access this list through model.attention_matrices
        # - Each module also stores its value matrix in model.value_matrices

        # First, ensure we have forward pass results
        _ = model(x)

        # Now we can extract the attention matrices and value matrices
        # from the model's temporal and feature channels

        # In a real implementation, you'd extract these from specific layers/modules
        attention_matrices = {}
        value_matrices = {}

        # Example for temporal channel
        attention_matrices['temporal'] = [m.attention_matrix for m in model.encoder.temporal_blocks]
        value_matrices['temporal'] = [m.value_matrix for m in model.encoder.temporal_blocks]

        # Example for feature channel
        attention_matrices['feature'] = [m.attention_matrix for m in model.encoder.feature_blocks]
        value_matrices['feature'] = [m.value_matrix for m in model.encoder.feature_blocks]

        return attention_matrices, value_matrices

    def compute_distillation_loss(self, x):
        """
        Compute the distillation loss between teacher and student models.

        Args:
            x: Input data

        Returns:
            Total distillation loss
        """
        # Get attention matrices and value matrices from both models
        teacher_att, teacher_val = self.get_attention_matrices(self.teacher_model, x)
        student_att, student_val = self.get_attention_matrices(self.student_model, x)

        # Initialize total loss
        total_loss = 0.0

        # Compute KL divergence loss for each channel
        for channel in ['temporal', 'feature']:
            channel_loss = 0.0

            # For each attention head (even though we use single-head, we loop for generality)
            for i in range(len(teacher_att[channel])):
                # Get teacher and student matrices
                teacher_attention = teacher_att[channel][i]
                student_attention = student_att[channel][i]

                teacher_value_rel = torch.matmul(teacher_val[channel][i],
                                                 teacher_val[channel][i].transpose(-2, -1))
                teacher_value_rel = F.softmax(teacher_value_rel / np.sqrt(teacher_value_rel.size(-1)), dim=-1)

                student_value_rel = torch.matmul(student_val[channel][i],
                                                 student_val[channel][i].transpose(-2, -1))
                student_value_rel = F.softmax(student_value_rel / np.sqrt(student_value_rel.size(-1)), dim=-1)

                # Compute KL divergence for attention scores
                attention_loss = F.kl_div(
                    F.log_softmax(student_attention, dim=-1),
                    F.softmax(teacher_attention, dim=-1),
                    reduction='batchmean'
                )

                # Compute KL divergence for value relationships
                value_rel_loss = F.kl_div(
                    F.log_softmax(student_value_rel, dim=-1),
                    F.softmax(teacher_value_rel, dim=-1),
                    reduction='batchmean'
                )

                # Add to channel loss
                channel_loss += attention_loss + value_rel_loss

            # Add to total loss
            total_loss += channel_loss

        return total_loss

    def adaptive_distillation_loss(self, x, test_loss):
        """
        Compute adaptive distillation loss weighted by the test loss.

        Args:
            x: Input data
            test_loss: Test loss of the teacher model

        Returns:
            Adaptive distillation loss
        """
        # Compute regular distillation loss
        distillation_loss = self.compute_distillation_loss(x)

        # Scale by the test loss (inverse relationship)
        # Lower test loss means better teacher, so we give more weight to distillation
        adaptive_factor = 1.0 / (test_loss + 1e-5)  # Add small epsilon to avoid division by zero

        # Normalize the factor to a reasonable range (optional)
        adaptive_factor = torch.clamp(adaptive_factor, 0.1, 10.0)

        return distillation_loss * adaptive_factor


class FederatedDevice:
    """
    Represents a single device in the federated learning setting.
    Each device has a local teacher model and a student model.
    """

    def __init__(self, model_class, model_params, device_id, device='cuda'):
        self.device_id = device_id
        self.device = device

        # Initialize teacher model (more complex)
        self.teacher_model = model_class(**model_params).to(device)

        # Initialize student model (simpler version)
        student_params = copy.deepcopy(model_params)
        student_params['num_blocks'] = max(1, model_params['num_blocks'] - 1)  # Reduce complexity
        self.student_model = model_class(**student_params).to(device)

        # Setup distillation mechanism
        self.distillation = AdaptiveAttentionDistillation(
            self.teacher_model, self.student_model, device
        )

        # Setup optimizers
        self.teacher_optimizer = optim.Adam(self.teacher_model.parameters(), lr=0.001)
        self.student_optimizer = optim.Adam(self.student_model.parameters(), lr=0.001)

    def local_training(self, train_loader, epochs=1, lambda_balance=0.5):
        """
        Perform local training of the teacher model.

        Args:
            train_loader: DataLoader for training data
            epochs: Number of training epochs
            lambda_balance: Balance coefficient for prediction and reconstruction losses

        Returns:
            Final test loss of the teacher model
        """
        self.teacher_model.train()
        final_test_loss = 0.0

        for epoch in range(epochs):
            epoch_loss = 0.0

            for batch_idx, (data, _) in enumerate(train_loader):
                data = data.to(self.device)

                # Forward pass
                self.teacher_optimizer.zero_grad()
                prediction, reconstruction = self.teacher_model(data)

                # Compute prediction loss (for next time step)
                pred_loss = F.mse_loss(prediction, data[:, -1, :])

                # Compute reconstruction loss
                recon_loss = F.mse_loss(reconstruction, data)

                # Combine losses using lambda
                loss = lambda_balance * pred_loss + (1 - lambda_balance) * recon_loss

                # Backward pass and optimization
                loss.backward()
                self.teacher_optimizer.step()

                epoch_loss += loss.item()

            # Compute test loss on the last batch for adaptive distillation
            with torch.no_grad():
                prediction, reconstruction = self.teacher_model(data)
                pred_loss = F.mse_loss(prediction, data[:, -1, :])
                recon_loss = F.mse_loss(reconstruction, data)
                final_test_loss = lambda_balance * pred_loss + (1 - lambda_balance) * recon_loss

        return final_test_loss

    def upload_distillation(self, train_loader, test_loss, epochs=1):
        """
        Perform knowledge distillation from teacher to student model (upload phase).

        Args:
            train_loader: DataLoader for training data
            test_loss: Test loss of the teacher model from local training
            epochs: Number of distillation epochs

        Returns:
            Student model state dict
        """
        self.teacher_model.eval()
        self.student_model.train()

        for epoch in range(epochs):
            for batch_idx, (data, _) in enumerate(train_loader):
                data = data.to(self.device)

                # Forward pass with student model
                self.student_optimizer.zero_grad()
                prediction, reconstruction = self.student_model(data)

                # Compute prediction and reconstruction losses
                pred_loss = F.mse_loss(prediction, data[:, -1, :])
                recon_loss = F.mse_loss(reconstruction, data)
                task_loss = pred_loss + recon_loss

                # Compute adaptive distillation loss
                distill_loss = self.distillation.adaptive_distillation_loss(data, test_loss)

                # Combine task loss and distillation loss
                total_loss = task_loss + distill_loss

                # Backward pass and optimization
                total_loss.backward()
                self.student_optimizer.step()

        # Return student model parameters for server aggregation
        return self.student_model.state_dict()

    def download_distillation(self, global_state_dict, train_loader, epochs=1):
        """
        Update student model with global state dict and distill knowledge back to teacher.

        Args:
            global_state_dict: Aggregated global model state dict
            train_loader: DataLoader for training data
            epochs: Number of distillation epochs
        """
        # Update student model with global parameters
        self.student_model.load_state_dict(global_state_dict)
        self.student_model.eval()
        self.teacher_model.train()

        # Create optimizer for teacher model for download distillation
        optimizer = optim.Adam(self.teacher_model.parameters(), lr=0.001)

        for epoch in range(epochs):
            for batch_idx, (data, _) in enumerate(train_loader):
                data = data.to(self.device)

                # Forward pass with teacher model
                optimizer.zero_grad()

                # Compute distillation loss (reversed roles)
                distill_loss = 0.0

                # Get attention matrices and value matrices from both models
                teacher_att, teacher_val = self.distillation.get_attention_matrices(self.teacher_model, data)
                student_att, student_val = self.distillation.get_attention_matrices(self.student_model, data)

                # Compute KL divergence loss for each channel
                for channel in ['temporal', 'feature']:
                    # For each attention head
                    for i in range(len(teacher_att[channel])):
                        # Get teacher and student matrices
                        teacher_attention = teacher_att[channel][i]
                        student_attention = student_att[channel][i]

                        teacher_value_rel = torch.matmul(teacher_val[channel][i],
                                                         teacher_val[channel][i].transpose(-2, -1))
                        teacher_value_rel = F.softmax(teacher_value_rel / np.sqrt(teacher_value_rel.size(-1)), dim=-1)

                        student_value_rel = torch.matmul(student_val[channel][i],
                                                         student_val[channel][i].transpose(-2, -1))
                        student_value_rel = F.softmax(student_value_rel / np.sqrt(student_value_rel.size(-1)), dim=-1)

                        # Note: the order is reversed from upload distillation
                        # Student is now the teacher

                        # Compute KL divergence for attention scores
                        attention_loss = F.kl_div(
                            F.log_softmax(teacher_attention, dim=-1),
                            F.softmax(student_attention, dim=-1),
                            reduction='batchmean'
                        )

                        # Compute KL divergence for value relationships
                        value_rel_loss = F.kl_div(
                            F.log_softmax(teacher_value_rel, dim=-1),
                            F.softmax(student_value_rel, dim=-1),
                            reduction='batchmean'
                        )

                        # Add to distillation loss
                        distill_loss += attention_loss + value_rel_loss

                # Backward pass and optimization
                distill_loss.backward()
                optimizer.step()


class FederatedServer:
    """
    Central server for federated learning coordination.
    """

    def __init__(self, model_class, model_params):
        self.model_class = model_class
        self.model_params = model_params

        # Student model parameters for global model
        student_params = copy.deepcopy(model_params)
        student_params['num_blocks'] = max(1, model_params['num_blocks'] - 1)

        # Initialize global model
        self.global_model = model_class(**student_params)

        # Keep track of devices
        self.devices = {}

    def add_device(self, device_id, device='cuda'):
        """Add a new device to the federated system."""
        self.devices[device_id] = FederatedDevice(
            self.model_class, self.model_params, device_id, device
        )

    def aggregate(self, device_models, selecte