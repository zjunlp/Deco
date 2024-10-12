import socket

import torch
import torch.distributed as dist
import os


DEVICE = None


def setup_dist(args):
    if dist.is_initialized():
        return

    if os.environ.get("MASTER_ADDR", None) is None:
        hostname = socket.gethostbyname(socket.getfqdn())
        os.environ["MASTER_ADDR"] = hostname
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        port = _find_free_port()
        os.environ["MASTER_PORT"] = str(port)

    dist.init_process_group("nccl")
    assert (
        args.batch_size % dist.get_world_size() == 0
    ), f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    global DEVICE
    DEVICE = device
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}, device={DEVICE}")


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def device():
    if not dist.is_initialized():
        raise NameError
    return DEVICE
