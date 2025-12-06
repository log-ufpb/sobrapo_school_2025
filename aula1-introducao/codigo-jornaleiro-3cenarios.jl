# ============================================================================ #
# ======================       Newsvendor Problem       ====================== #
# ============================================================================ #
# Open REPL in VS Code -> Ctrl+Shift+p

using Distributions, Random
using JuMP
using GLPK              # Solver Gratuito de Programação Linear & Inteira Mista - https://github.com/jump-dev/GLPK.jl | https://www.gnu.org/software/glpk/

using Gurobi            # Solver Comercial de Programação Matemática - https://github.com/jump-dev/Gurobi.jl | https://www.gurobi.com/
                        #   --> A PUC-Rio possui licença acadêmica gratuita para os alunos.
#

# ======================     Parâmetros do Problema   ======================== #

u = 300;
q = 40;
r = 15;
c = 25;

# ============================================================================ #

# ==============     Cenários e Probabilidades de Demanda     ================ #

d1 = 150;
p1 = 0.55;

d2 = 200;
p2 = 0.30;

d3 = 250;
p3 = 0.15;

# ============================================================================ #

# ========================     Sample Problem     ============================ #

#NewsVendorProb = Model(GLPK.Optimizer);
NewsVendorProb = Model(Gurobi.Optimizer);

# ========== Variáveis de Decisão ========== #

# -----> Decisão de 1o Estágio

@variable(NewsVendorProb, x >= 0);

# -----> Decisão de 2o Estágio - Venda Jornais

@variable(NewsVendorProb, y1 >= 0);
@variable(NewsVendorProb, y2 >= 0);
@variable(NewsVendorProb, y3 >= 0);

# -----> Decisão de 2o Estágio - Re-Venda Jornais

@variable(NewsVendorProb, z1 >= 0);
@variable(NewsVendorProb, z2 >= 0);
@variable(NewsVendorProb, z3 >= 0);

# -----> Receita (Lucro) Total

@variable(NewsVendorProb, R1);
@variable(NewsVendorProb, R2);
@variable(NewsVendorProb, R3);

# ========== Restrições ========== #

# -----> Restrição de 1o Estágio - Capacidade Máxima de Compra

@constraint(NewsVendorProb, x <= u);

# -----> Restrição de 2o Estágio - Venda Limitada pela Demanda (por cenário)

@constraint(NewsVendorProb, y1 <= d1);
@constraint(NewsVendorProb, y2 <= d2);
@constraint(NewsVendorProb, y3 <= d3);

# -----> Restrição de 2o Estágio - Venda & Re-Venda Limitada pela Compra (por cenário)

@constraint(NewsVendorProb, y1 + z1 <= x);
@constraint(NewsVendorProb, y2 + z2 <= x);
@constraint(NewsVendorProb, y3 + z3 <= x);

# -----> Calculo da Receita Total (por cenário)

@constraint(NewsVendorProb, R1 == q*y1 + r*z1 - c*x);
@constraint(NewsVendorProb, R2 == q*y2 + r*z2 - c*x);
@constraint(NewsVendorProb, R3 == q*y3 + r*z3 - c*x);

# ========== Função Objetivo ========== #

@objective(NewsVendorProb, Max, p1*R1 + p2*R2 + p3*R3);

optimize!(NewsVendorProb);

status      = termination_status(NewsVendorProb);

RecTot      = JuMP.objective_value(NewsVendorProb);
xOpt        = JuMP.value.(x);

println("\n");
println("=================================================================================================\n");
println("Status: ", status);
println("Lucro Jornaleiro: ", RecTot);
println("Quant. Jornais: ", xOpt, "\n");

println("Cenário 1 -- Demanda: ", d1, " -- Venda: ", JuMP.value.(y1), " -- Re-Venda -- ", JuMP.value.(z1));
println("Cenário 2 -- Demanda: ", d2, " -- Venda: ", JuMP.value.(y2), " -- Re-Venda -- ", JuMP.value.(z2));
println("Cenário 3 -- Demanda: ", d3, " -- Venda: ", JuMP.value.(y3), " -- Re-Venda -- ", JuMP.value.(z3));

println("\n=================================================================================================");