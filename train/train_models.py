from models.regret_net import RegretNet
from models.policy_net import PolicyNet


def train_models(regret_net: RegretNet = None,
                 policy_net: PolicyNet = None,
                 advantage_buffer=None,
                 policy_buffer=None,
                 epochs: int = 5) -> dict:
    """
    Standalone training helper for RegretNet and PolicyNet.

    FIX: was just a print statement placeholder.
    Now delegates to DeepCFRTrainer helpers when a trainer is passed,
    or can be used directly with pre-built buffers.
    """
    # If buffers not provided, return early with a warning
    if advantage_buffer is None or policy_buffer is None:
        print("[train_models] No buffers provided — skipping training.")
        return {"regret_loss": None, "policy_loss": None}

    from train.train_deep_cfr import DeepCFRTrainer

    # Create a minimal trainer to reuse training logic
    class _DummyEnv:
        def reset(self): return {}
        def get_legal_actions(self): return ["fold", "call", "raise"]
        def step(self, a): return {}, 0.0, True, {}

    trainer = DeepCFRTrainer(_DummyEnv())
    trainer.regret_net = regret_net or trainer.regret_net
    trainer.policy_net = policy_net or trainer.policy_net
    trainer.advantage_buffer = advantage_buffer
    trainer.policy_buffer = policy_buffer

    r_loss = trainer.train_regret_net(epochs=epochs)
    p_loss = trainer.train_policy_net(epochs=epochs)

    print(f"[train_models] RegretNet loss={r_loss:.6f} | PolicyNet loss={p_loss:.6f}")
    return {"regret_loss": r_loss, "policy_loss": p_loss}
