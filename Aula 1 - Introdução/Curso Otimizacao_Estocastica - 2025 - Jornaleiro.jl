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

# ========================     Sampling Process     ========================== #

nCenarios = 20000;                     # Number of Scenarios
Ω = 1:nCenarios;                    # Set of Scenarios
p = ones(nCenarios)*(1/nCenarios);  # Equal Probability

dmin = 100;
dmax = 300;

# ===================================
#      =====> Using Julia <=====     
# ===================================

# -> https://juliastats.org/Distributions.jl/stable/ <- #

#Random.seed!(1);
d  = rand(Uniform(dmin, dmax), nCenarios);

# ============================================================================ #

# ========================     Sample Problem     ============================ #

#NewsVendorProb = Model(GLPK.Optimizer);
NewsVendorProb = Model(Gurobi.Optimizer);

# ========== Variáveis de Decisão ========== #

@variable(NewsVendorProb, x >= 0);
@variable(NewsVendorProb, y[Ω] >= 0);
@variable(NewsVendorProb, z[Ω] >= 0);
@variable(NewsVendorProb, R[Ω]);

# ========== Restrições ========== #

@constraint(NewsVendorProb, Rest1, x <= u);
@constraint(NewsVendorProb, Rest2[ω in Ω], y[ω] <= d[ω]);
@constraint(NewsVendorProb, Rest3[ω in Ω], y[ω] + z[ω] <= x);
@constraint(NewsVendorProb, Rest4[ω in Ω], R[ω] == q*y[ω] + r*z[ω] - c*x);

# ========== Função Objetivo ========== #

@objective(NewsVendorProb, Max, sum(R[ω]*p[ω] for ω in Ω));

optimize!(NewsVendorProb);

status      = termination_status(NewsVendorProb);

Profit      = JuMP.objective_value(NewsVendorProb);
xOpt        = JuMP.value.(x);
yOpt        = JuMP.value.(y);
zOpt        = JuMP.value.(z);

println("\n");
println("========================================\n");
println("Status: ", status);
println("Lucro Jornaleiro: ", Profit);
println("Quant. Jornais: ", xOpt);
println("\n========================================");