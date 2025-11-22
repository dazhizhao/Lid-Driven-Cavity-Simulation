import numpy as np
import ufl
import basix.ufl
from mpi4py import MPI
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
from petsc4py import PETSc
from tqdm import tqdm

# 参数设置
N = 41
dt = 0.01
num_steps = 1000
nu = 0.1

comm = MPI.COMM_WORLD
msh = mesh.create_unit_square(comm, N, N, mesh.CellType.triangle)

# 选用 Taylor-Hood 以满足 LBB 稳定性
v_cg2 = basix.ufl.element("Lagrange", "triangle", 2, shape=(msh.geometry.dim,))
p_cg1 = basix.ufl.element("Lagrange", "triangle", 1)
V = fem.functionspace(msh, v_cg2)
Q = fem.functionspace(msh, p_cg1)

# 边界条件定义
fdim = msh.topology.dim - 1

# 1. 速度边界：壁面无滑移，顶盖驱动
wall_dofs = fem.locate_dofs_topological(V, fdim, mesh.locate_entities_boundary(msh, fdim, 
    lambda x: np.logical_or(np.isclose(x[0], 0), np.logical_or(np.isclose(x[0], 1), np.isclose(x[1], 0)))))
lid_dofs = fem.locate_dofs_topological(V, fdim, mesh.locate_entities_boundary(msh, fdim, 
    lambda x: np.isclose(x[1], 1)))

bc_wall = fem.dirichletbc(np.array([0.0, 0.0], dtype=PETSc.ScalarType), wall_dofs, V)
bc_lid = fem.dirichletbc(np.array([1.0, 0.0], dtype=PETSc.ScalarType), lid_dofs, V)
bcs_u = [bc_wall, bc_lid]

# 2. 压力边界：固定左下角点为 0，消除纯 Neumann 边界导致的零空间
corner_dofs = fem.locate_dofs_geometrical(Q, lambda x: np.logical_and(np.isclose(x[0], 0), np.isclose(x[1], 0)))
bc_p = fem.dirichletbc(PETSc.ScalarType(0), corner_dofs, Q)
bcs_p = [bc_p]

# 变分形式 (Chorin 分步投影法)
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
p, q = ufl.TrialFunction(Q), ufl.TestFunction(Q)

u_n = fem.Function(V)     # 上一时刻
u_star = fem.Function(V)  # 预测速度
u_new = fem.Function(V)   # 修正后速度
p_new = fem.Function(Q)   # 新压力

k = fem.Constant(msh, PETSc.ScalarType(dt))
nu_const = fem.Constant(msh, PETSc.ScalarType(nu))
f = fem.Constant(msh, np.array([0.0, 0.0], dtype=PETSc.ScalarType))

# 1: 动量方程 (对流项显式线性化，扩散项隐式)
F1 = (1.0/k)*ufl.inner(u - u_n, v)*ufl.dx + \
     ufl.inner(ufl.dot(ufl.grad(u_n), u_n), v)*ufl.dx + \
     nu_const*ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx - \
     ufl.inner(f, v)*ufl.dx
a1, L1 = ufl.lhs(F1), ufl.rhs(F1)

# 2: 压力泊松方程
F2 = ufl.inner(ufl.grad(p), ufl.grad(q))*ufl.dx + (1.0/k)*ufl.div(u_star)*q*ufl.dx
a2, L2 = ufl.lhs(F2), ufl.rhs(F2)

# 3: 速度修正
F3 = ufl.inner(u - u_star, v)*ufl.dx + k*ufl.inner(ufl.grad(p_new), v)*ufl.dx
a3, L3 = ufl.lhs(F3), ufl.rhs(F3)

# 组装矩阵 (由于 dt 恒定，矩阵只需组装一次)
A1 = assemble_matrix(fem.form(a1), bcs=bcs_u); A1.assemble()
A2 = assemble_matrix(fem.form(a2), bcs=bcs_p); A2.assemble()
A3 = assemble_matrix(fem.form(a3), bcs=bcs_u); A3.assemble()

b1 = assemble_vector(fem.form(L1))
b2 = assemble_vector(fem.form(L2))
b3 = assemble_vector(fem.form(L3))

# 求解器
def get_solver(A):
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    return solver

s1, s2, s3 = get_solver(A1), get_solver(A2), get_solver(A3)

# Paraview输出设置
xdmf = io.XDMFFile(comm, "results_cavity.xdmf", "w")
xdmf.write_mesh(msh)
# 用于可视化输出的 P1 空间
V_out = fem.functionspace(msh, basix.ufl.element("Lagrange", "triangle", 1, shape=(msh.geometry.dim,)))
u_out = fem.Function(V_out)
u_out.name = "Velocity"
p_new.name = "Pressure"

t = 0.0
for i in tqdm(range(num_steps), desc="Processing"):
    t += dt
    
    # 1. 预测
    with b1.localForm() as loc: loc.set(0)
    assemble_vector(b1, fem.form(L1))
    apply_lifting(b1, [fem.form(a1)], [bcs_u])
    b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b1, bcs_u)
    s1.solve(b1, u_star.x.petsc_vec)
    u_star.x.scatter_forward()

    # 2. 压力
    with b2.localForm() as loc: loc.set(0)
    assemble_vector(b2, fem.form(L2))
    apply_lifting(b2, [fem.form(a2)], [bcs_p])
    b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b2, bcs_p)
    s2.solve(b2, p_new.x.petsc_vec)
    p_new.x.scatter_forward()

    # 3. 修正
    with b3.localForm() as loc: loc.set(0)
    assemble_vector(b3, fem.form(L3))
    apply_lifting(b3, [fem.form(a3)], [bcs_u])
    b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b3, bcs_u)
    s3.solve(b3, u_new.x.petsc_vec)
    u_new.x.scatter_forward()

    # 更新
    u_n.x.array[:] = u_new.x.array[:]
    
    u_out.interpolate(u_new)
    xdmf.write_function(u_out, t)
    xdmf.write_function(p_new, t)

xdmf.close()