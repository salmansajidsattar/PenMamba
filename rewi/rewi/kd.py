import torch


def kd_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    lengths: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    '''Frame-level distillation loss: KL(teacher || student), averaged over valid frames.

    Args:
        student (torch.Tensor): Student probabilities (size_batch, len_seq, num_cls).
        teacher (torch.Tensor): Teacher probabilities, same shape.
        lengths (torch.Tensor): Number of valid frames per sample (size_batch).
        eps (float, optional): Floor for the logs. Defaults to 1e-6.

    Returns:
        torch.Tensor: Scalar loss.
    '''
    student = student.float()
    teacher = teacher.float()
    kl = (
        teacher
        * (teacher.clamp_min(eps).log() - student.clamp_min(eps).log())
    ).sum(dim=-1)  # (size_batch, len_seq)
    steps = torch.arange(kl.shape[1], device=kl.device)
    mask = steps[None, :] < lengths.to(kl.device)[:, None]

    return (kl * mask).sum() / mask.sum().clamp_min(1)
