class Req:
    def __init__(self, input_ids, cached_len, output_len):
        self.input_ids = list(input_ids)
        self.cached_len = cached_len
        self.output_len = output_len
        self.device_len = len(self.input_ids)
        self.max_device_len = len(self.input_ids) + output_len
        if not (output_len >= 0 and 0 <= cached_len < self.device_len <= self.max_device_len):
            raise ValueError(f"非法的长度：cached_len={cached_len}, device_len={self.device_len}, "
                             f"max_device_len={self.max_device_len}")

    @property
    def remain_len(self):
        return self.max_device_len - self.device_len

    @property
    def extend_len(self):
        return self.device_len - self.cached_len

    @property
    def can_decode(self):
        return self.remain_len > 0

    def complete_one(self):
        self.cached_len = self.device_len
        self.device_len += 1

    def append_host(self, token):
        self.input_ids.append(token)
