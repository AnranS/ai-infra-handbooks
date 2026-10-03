def gemm_intensity(M, N, K, bytes_per_elem=2):
    pass


def roofline_time(flops, bytes_, peak_tflops, bandwidth_gbs):
    pass


def bound(M, N, K, peak_tflops, bandwidth_gbs, bytes_per_elem=2):
    pass


def decode_step_ms(num_params, batch, peak_tflops, bandwidth_gbs, bytes_per_param=2):
    pass
