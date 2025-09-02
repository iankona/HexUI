from ortools.linear_solver import pywraplp

def LinearProgrammingExample():
    """线性规划示例。"""
    # 创建一个GLOP求解器，命名为LinearExample。
    solver = pywraplp.Solver.CreateSolver("GLOP")
    if not solver:
        return

    # 创建两个变量x和y，允许它们取非负值。
    x = solver.NumVar(0, solver.infinity(), "x")
    y = solver.NumVar(0, solver.infinity(), "y")

    print("变量个数 =", solver.NumVariables())

    # 约束条件0: x + 2y <= 14。
    solver.Add(x + 2 * y <= 14.0)

    # 约束条件1: 3x - y >= 0。
    solver.Add(3 * x - y >= 0.0)

    # 约束条件2: x - y <= 2。
    solver.Add(x - y <= 2.0)

    print("约束条件个数 =", solver.NumConstraints())

    # 目标函数: 最大化3x + 4y。
    solver.Maximize(3 * x + 4 * y)

    # 求解问题。
    print(f"使用 {solver.SolverVersion()} 求解")
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        print("解:")
        print(f"目标值 = {solver.Objective().Value():0.1f}")
        print(f"x = {x.solution_value():0.1f}")
        print(f"y = {y.solution_value():0.1f}")
    else:
        print("该问题没有最优解。")

    print("\n高级用法:")
    print(f"问题求解时间: {solver.wall_time():d} 毫秒")
    print(f"问题求解迭代次数: {solver.iterations():d} 次")

# 调用函数求解线性规划问题
LinearProgrammingExample()
