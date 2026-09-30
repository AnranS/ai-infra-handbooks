from checker import check
from solution import min_path_sum, paths_with_obstacles, unique_paths


def test_example():
    check(unique_paths(3, 7), 28, "3x7 网格")
    check(paths_with_obstacles([[0, 0, 0], [0, 1, 0], [0, 0, 0]]), 2, "中间有障碍")
    check(min_path_sum([[1, 3, 1], [1, 5, 1], [4, 2, 1]]), 7, "最小代价")


def test_paths_edges():
    check(unique_paths(1, 1), 1, "只有一个格子")
    check(unique_paths(1, 5), 1, "只能一直往右")
    check(unique_paths(5, 1), 1, "只能一直往下")
    check(unique_paths(0, 5), 0, "没有网格")
    check(unique_paths(2, 2), 2, "两条路")


def test_obstacles_edges():
    check(paths_with_obstacles([]), 0, "空")
    check(paths_with_obstacles([[1]]), 0, "起点是障碍")
    check(paths_with_obstacles([[0]]), 1, "单格")
    check(paths_with_obstacles([[0, 1], [0, 0]]), 1, "绕开障碍")
    check(paths_with_obstacles([[0, 1], [1, 0]]), 0, "堵死了")


def test_min_path_edges():
    check(min_path_sum([]), 0, "空")
    check(min_path_sum([[5]]), 5, "单格")
    check(min_path_sum([[1, 2, 3]]), 6, "单行")
    check(min_path_sum([[1], [2], [3]]), 6, "单列")


def test_min_path_choice():
    check(min_path_sum([[1, 100], [1, 1]]), 3, "绕开大数")
    check(min_path_sum([[1, 1], [100, 1]]), 3, "另一个方向")


def test_agreement():
    # 没有障碍时，障碍版本应该和 unique_paths 一致
    for m, n in ((1, 1), (2, 3), (4, 4), (3, 6)):
        grid = [[0] * n for _ in range(m)]
        check(paths_with_obstacles(grid), unique_paths(m, n), f"{m}x{n} 无障碍时一致")


def test_large():
    check(unique_paths(20, 20), 35345263800, "20x20")
    grid = [[1] * 200 for _ in range(200)]
    check(min_path_sum(grid), 399, "200x200 全是 1：走 399 步")
