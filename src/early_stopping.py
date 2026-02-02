import torch
import torch.nn as nn


class EarlyStopping:
    """Implement the Automatic Early Stopping technique (Lutz Prechelt, 1998).

    In particular, it uses the GL_alpha criterion: the training stops
    when the Generalization Loss is greater than the alpha value.
    """

    def __init__(self, alpha: float = 5.0, path: str = "checkpoint.pth") -> None:
        """Initialize monitoring.

        Args:
            alpha: Generalization Loss percentage threshold (e.g., 5.0).
            path: Path to save the best model (E_opt).

        """
        self.alpha: float = alpha
        self.path: str = path
        self.min_v_loss: float = float("inf")
        self.best_epoch: int = 0
        self.stop: bool = False

    def __call__(self, v_loss: float, epoch: int, model: nn.Module) -> None:
        """Check the stopping condition.

        Args:
            v_loss: Validation loss of the current epoch.
            epoch: Index of the current epoch.
            model: The model to save in case of improvement.

        """
        if v_loss < self.min_v_loss:
            self.min_v_loss = v_loss
            self.best_epoch = epoch
            # Save the "optimal" model (E_opt) mentioned in the paper
            torch.save(model.state_dict(), self.path)

        # GL(t) = 100 * (E_va(t) / E_opt(t) - 1)
        gl_t = 100 * (v_loss / self.min_v_loss - 1)

        if gl_t > self.alpha:
            print(f"\n[Early Stopping] GL: {gl_t:.2f}% > Alpha: {self.alpha}%")
            self.stop = True
