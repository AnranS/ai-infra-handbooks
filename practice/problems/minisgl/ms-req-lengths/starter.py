class Req:
    def __init__(self, input_ids, cached_len, output_len):
        self.input_ids = list(input_ids)
        self.cached_len = cached_len
        self.output_len = output_len
        # TODO：device_len、max_device_len 与合法性检查
