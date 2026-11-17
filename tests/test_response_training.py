from pondlite.train import SFTDataset


class BoundaryTokenizer:
    def __call__(self, text, **kwargs):
        if "<|endoftext|>" in text:
            return {"input_ids": [10, 11, 12, 13]}
        return {"input_ids": [10, 11]}


def test_sft_loss_masks_prompt_tokens():
    record = {"instruction": "Question", "input": None, "output": "Answer"}
    example = SFTDataset([record], BoundaryTokenizer())[0]
    assert example["input_ids"].tolist() == [10, 11, 12, 13]
    assert example["labels"].tolist() == [-100, -100, 12, 13]
