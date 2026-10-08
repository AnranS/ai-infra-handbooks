import torch


def collate(batch):
    lens = [len(ids) for ids, _ in batch]
    n = max(lens)
    ids = torch.zeros(len(batch), n, dtype=torch.long)
    mask = torch.zeros(len(batch), n, dtype=torch.bool)
    for i, (seq, _) in enumerate(batch):
        ids[i, : lens[i]] = seq
        mask[i, : lens[i]] = True
    return {"ids": ids, "mask": mask,
            "labels": torch.tensor([label for _, label in batch], dtype=torch.long)}


def bucket_batches(lengths, batch_size):
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    return [order[i : i + batch_size] for i in range(0, len(order), batch_size)]


def pad_fraction(lengths, batches):
    cells = sum(len(b) * max(lengths[i] for i in b) for b in batches)
    used = sum(lengths[i] for b in batches for i in b)
    return (cells - used) / cells
