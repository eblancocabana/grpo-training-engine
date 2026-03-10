def _make_fake_tokenizer():
    class FakeTokenizer:
        def __call__(
            self,
            texts,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_tensors=None,
        ):
            if isinstance(texts, str):
                texts = [texts]
            return {"input_ids": [text.split() for text in texts]}

        def apply_chat_template(
            self, messages, tokenize=False, add_generation_prompt=True
        ):
            return messages[0]["content"]

    return FakeTokenizer()


def test_compute_question_length_percentile_from_list():
    from src.data import gsm8k_loader

    dataset = [
        {"question": "one"},
        {"question": "one two"},
        {"question": "one two three"},
        {"question": "one two three four"},
        {"question": "one two three four five six seven eight nine ten"},
    ]
    tokenizer = _make_fake_tokenizer()

    percentile_length = gsm8k_loader._compute_question_length_percentile(
        dataset, tokenizer, percentile=95.0, batch_size=2
    )

    assert percentile_length == 8


def test_dataset_updates_max_prompt_length(monkeypatch):
    from src.data import gsm8k_loader

    fake_dataset = [
        {"question": "one"},
        {"question": "one two three"},
        {"question": "one two three four five six seven eight"},
    ]
    tokenizer = _make_fake_tokenizer()

    monkeypatch.setattr(
        gsm8k_loader, "load_dataset", lambda *args, **kwargs: fake_dataset
    )

    dataset = gsm8k_loader.GRPOGSM8KDataset(
        tokenizer=tokenizer, split="train", max_prompt_length=512
    )

    assert dataset.max_prompt_length == 7
