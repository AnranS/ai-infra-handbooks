def gemm_intensity(M, N, K, bytes_per_elem=2):
    return 2 * M * N * K / ((M * K + K * N + M * N) * bytes_per_elem)


def roofline_time(flops, bytes_, peak_tflops, bandwidth_gbs):
    return max(flops / (peak_tflops * 1e12), bytes_ / (bandwidth_gbs * 1e9))


def bound(M, N, K, peak_tflops, bandwidth_gbs, bytes_per_elem=2):
    ridge = peak_tflops * 1e12 / (bandwidth_gbs * 1e9)
    return "compute" if gemm_intensity(M, N, K, bytes_per_elem) >= ridge else "memory"


def decode_step_ms(num_params, batch, peak_tflops, bandwidth_gbs, bytes_per_param=2):
    return roofline_time(2 * num_params * batch, num_params * bytes_per_param, peak_tflops, bandwidth_gbs) * 1e3
