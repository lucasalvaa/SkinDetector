import torch
import torch.nn as nn


class EarlyStopping:
    """Implement the Automatic Early Stopping technique (Lutz Prechelt, 1998)
    In particular, it uses the GL_alpha criterion: the training stops
    when the Generalization Loss is greater than the alpha value.
    """

    def __init__(self, alpha: float = 5.0, path: str = "checkpoint.pth") -> None:
        """Inizializza il monitoraggio.

        Args:
            alpha: Soglia percentuale di Generalization Loss (es. 5.0).
            path: Percorso dove salvare il miglior modello (E_opt).

        """
        self.alpha: float = alpha
        self.path: str = path
        self.min_v_loss: float = float("inf")
        self.best_epoch: int = 0
        self.stop: bool = False

    def __call__(self, v_loss: float, epoch: int, model: nn.Module) -> None:
        """Verifica la condizione di arresto.

        Args:
            v_loss: Loss di validazione dell'epoca corrente.
            epoch: Indice dell'epoca attuale.
            model: Il modello da salvare in caso di miglioramento.

        """
        if v_loss < self.min_v_loss:
            self.min_v_loss = v_loss
            self.best_epoch = epoch
            # Salviamo il modello "ottimale" (E_opt) citato nel paper
            torch.save(model.state_dict(), self.path)

        # GL(t) = 100 * (E_va(t) / E_opt(t) - 1)
        gl_t = 100 * (v_loss / self.min_v_loss - 1)

        if gl_t > self.alpha:
            print(f"\n[Early Stopping] GL: {gl_t:.2f}% > Alpha: {self.alpha}%")
            self.stop = True
