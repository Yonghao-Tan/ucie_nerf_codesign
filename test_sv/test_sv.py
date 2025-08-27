def greedy_algorithm(total_sum=24, weights=[6, 5, 4, 3]):
    # 初始化变量
    a, b, c, d = 0, 0, 0, 0
    remaining = total_sum

    # 定义误差函数
    def error(a, b, c, d):
        return abs(a * weights[0] - b * weights[1]) + \
               abs(b * weights[1] - c * weights[2]) + \
               abs(c * weights[2] - d * weights[3])

    # 贪心分配
    while remaining > 0:
        # 尝试将剩余值分配给每个变量，选择使误差最小的分配
        best_error = float('inf')
        best_choice = None

        for i in range(4):
            # 分别尝试增加a, b, c, d
            new_a, new_b, new_c, new_d = a, b, c, d
            if i == 0:
                new_a += 1
            elif i == 1:
                new_b += 1
            elif i == 2:
                new_c += 1
            elif i == 3:
                new_d += 1

            # 计算新的误差
            current_error = error(new_a, new_b, new_c, new_d)

            # 更新最优解
            if current_error < best_error:
                best_error = current_error
                best_choice = (new_a, new_b, new_c, new_d)

        # 更新分配结果
        a, b, c, d = best_choice
        remaining -= 1

    # 打印结果和均衡情况
    print("Solution: a =", a, ", b =", b, ", c =", c, ", d =", d)
    values = [a * weights[0], b * weights[1], c * weights[2], d * weights[3]]
    print("Balanced Values: a*s1 =", values[0], ", b*s2 =", values[1], 
          ", c*s3 =", values[2], ", d*s4 =", values[3])
    print("Differences: |a*s1 - b*s2| =", abs(values[0] - values[1]),
          ", |b*s2 - c*s3| =", abs(values[1] - values[2]),
          ", |c*s3 - d*s4| =", abs(values[2] - values[3]))

    return a, b, c, d


# 调用函数
solution = greedy_algorithm(weights=[6, 5, 4, 3])
solution = greedy_algorithm(weights=[6, 6, 4, 4])
solution = greedy_algorithm(weights=[5, 4, 3, 2])
solution = greedy_algorithm(weights=[5, 4, 4, 3])
solution = greedy_algorithm(weights=[6, 5, 4, 4])