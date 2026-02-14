import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class TestIntegration:
    """End-to-end integration tests."""

    def setup_fake_bnb(self):
        import sys, types

        fake_bnb = types.ModuleType("bitsandbytes")
        fake_bnb.nn = types.SimpleNamespace()
        fake_bnb.nn.Linear4bit = None
        sys.modules["bitsandbytes"] = fake_bnb

    def test_training_step_end_to_end(self):
        self.setup_fake_bnb()
        from src.grpo.algorithm import GRPOTrainer
        from src.grpo.verifier import RuleBasedVerifier

        device = torch.device("cuda")

        class TinyPolicy(nn.Module):
            def __init__(self, vocab=8, hidden=16):
                super().__init__()
                self.embed = nn.Embedding(vocab, hidden)
                self.fc = nn.Linear(hidden, vocab)

            def forward(self, input_ids):
                x = self.embed(input_ids)
                return self.fc(x)

        model = TinyPolicy().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        trainer = GRPOTrainer(group_size=2)
        verifier = RuleBasedVerifier()
        prompt_ids = torch.randint(0, 8, (2, 4), device=device)
        with torch.no_grad():
            logits = model(prompt_ids)
            probs = F.softmax(logits, dim=-1)
            generated = torch.argmax(probs, dim=-1)
        responses = [f"\\boxed{x.item()}" for x in generated[:, -1]]
        rewards_list = []
        for r in responses:
            result = verifier.verify(r, "5")
            rewards_list.append(result[0])
        rewards = torch.tensor(rewards_list, device=device)
        advantages = trainer.calculate_advantages(rewards, group_size=2)
        old_logits = logits.detach()
        for step in range(5):
            optimizer.zero_grad()
            new_logits = model(prompt_ids)
            loss, metrics = trainer.compute_grpo_loss(
                new_logits,
                advantages,
                old_policy_logits=old_logits,
                target_ids=generated,
            )
            loss.backward()
            optimizer.step()
        assert loss.item() is not None
        assert "ratio_mean" in metrics

    def test_overfitting_tiny_dataset(self):
        from src.grpo.algorithm import GRPOTrainer

        device = torch.device("cuda")

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 4)

            def forward(self, x):
                return self.fc(x)

        model = TinyModel().to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        trainer = GRPOTrainer(group_size=1)
        x = torch.randn(1, 4, device=device)
        target = torch.tensor([0], device=device)
        initial_logits = model(x)
        initial_loss = F.cross_entropy(initial_logits, target)
        for _ in range(50):
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
        final_logits = model(x)
        final_loss = F.cross_entropy(final_logits, target)
        assert final_loss < initial_loss

    def test_gradient_accumulation_simulation(self):
        device = torch.device("cuda")
        model = nn.Linear(4, 4).to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        accumulation_steps = 4
        accumulated_grad = None
        for i in range(accumulation_steps):
            x = torch.randn(1, 4, device=device)
            y = torch.randint(0, 4, (1,), device=device)
            logits = model(x)
            loss = F.cross_entropy(logits, y) / accumulation_steps
            loss.backward()
            if accumulated_grad is None:
                accumulated_grad = model.weight.grad.clone()
            else:
                accumulated_grad += model.weight.grad
        assert accumulated_grad is not None
        assert not torch.allclose(accumulated_grad, torch.zeros_like(accumulated_grad))
