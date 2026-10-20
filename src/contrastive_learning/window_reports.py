"""Reports paired with the exact jittered/cached window reads used by run 5."""

from torch.utils.data import Dataset


class WindowReportDataset(Dataset):
    def __init__(self, records, config, *, training, label_table=None):
        # Lazy import: fine-tuning also depends on the contrastive data modules.
        from .finetune import WindowDataset

        self.records = list(records)
        settings = {**config.image_kwargs, "windows_per_plane": config.windows_per_plane,
                    "window_cache_dir": str(config.window_cache_dir),
                    # The run-5 window reader takes the fine-tuning config's {"to": path} shape.
                    "train_series_mirror": ({"to": str(config.train_series_mirror)}
                                            if config.train_series_mirror is not None else None)}
        settings.pop("windows", None)
        self.windows = WindowDataset(self.records, settings, training=training)
        self.label_table = label_table

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        item = self.windows[index]
        item["report"] = self.records[index].report
        if self.label_table is not None:
            values, mask = self.label_table.get(item["study_uid"])
            item["label_values"], item["label_mask"] = values, mask
        return item
