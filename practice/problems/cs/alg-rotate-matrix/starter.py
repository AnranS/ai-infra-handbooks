def rotate(matrix):
    n = len(matrix)
    for i in range(n):
        for j in range(n):                     # 遍历整个矩阵：每对元素被交换两次，等于没换
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    for row in matrix:
        row.reverse()


def spiral(matrix):
    if not matrix:
        return []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1
    out = []
    while top <= bottom and left <= right:
        for j in range(left, right + 1):
            out.append(matrix[top][j])
        top += 1
        for i in range(top, bottom + 1):
            out.append(matrix[i][right])
        right -= 1
        for j in range(right, left - 1, -1):   # 没检查 top <= bottom：单行矩阵会重复
            out.append(matrix[bottom][j])
        bottom -= 1
        for i in range(bottom, top - 1, -1):
            out.append(matrix[i][left])
        left += 1
    return out
