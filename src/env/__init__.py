# Lazy import — avoids circular dependency when importing env.spaces or other
# submodules before F110ParallelEnv is fully initialized.

def __getattr__(name: str):
    if name == "F110ParallelEnv":
        from env.f110ParallelEnv import F110ParallelEnv
        return F110ParallelEnv
    raise AttributeError(f"module 'env' has no attribute {name!r}")
