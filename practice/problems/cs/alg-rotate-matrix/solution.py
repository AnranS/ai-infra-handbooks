def rotate(matrix):
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):              # 只遍历上三角
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    for row in matrix:
        row.reverse()


def spiral(matrix):
    if not matrix or not matrix[0]:
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
        if top <= bottom:                      # 还有行没走完才往回走
            for j in range(right, left - 1, -1):
                out.append(matrix[bottom][j])
            bottom -= 1
        if left <= right:
            for i in range(bottom, top - 1, -1):
                out.append(matrix[i][left])
            left += 1
    return out
