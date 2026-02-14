from torch.utils.data import Dataset, DataLoader


def test_dataloader_batching():
    class FakeDataset(Dataset):
        def __init__(self, size=10):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
                "question": f"Q{idx}",
                "answer": f"A{idx}",
            }

    def collate_fn(batch):
        return {
            "input_ids": [b["input_ids"] for b in batch],
            "questions": [b["question"] for b in batch],
            "answers": [b["answer"] for b in batch],
        }

    dataset = FakeDataset()
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    batch = next(iter(loader))
    assert len(batch["questions"]) == 2
    assert "answers" in batch
