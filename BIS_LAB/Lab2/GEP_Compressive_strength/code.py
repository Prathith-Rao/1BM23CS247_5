import random
import numpy as np
import math

# ---------------- DATASET ----------------
dataset = np.array([
    [350, 100, 180, 35],
    [300, 120, 150, 28],
    [400, 90, 200, 42],
    [380, 110, 190, 38],
    [360, 105, 180, 36],
])

variables = ['C', 'F', 'W']
functions = ['+', '-', '*', '/', 'sin', 'cos', 'exp', 'log', 'sqrt']

POP_SIZE = 50
GENS = 50
MUTATION_RATE = 0.2
HEAD_LENGTH = 5
TAIL_LENGTH = HEAD_LENGTH * (len(functions)-1) + 1
NUM_GENES = 3  # More genes for structured combination

# ---------------- HELPER FUNCTIONS ----------------
def generate_gene():
    head = [random.choice(functions + variables) for _ in range(HEAD_LENGTH)]
    tail = [random.choice(variables) for _ in range(TAIL_LENGTH)]
    return head + tail

def gene_to_expression(gene):
    stack = []
    for symbol in reversed(gene):
        if symbol in variables:
            stack.append(symbol)
        elif symbol in functions:
            if symbol in ['sin', 'cos', 'exp', 'log', 'sqrt']:
                if stack:
                    a = stack.pop()
                    stack.append(f"{symbol}({a})")
            else:  # binary operator
                if len(stack) >= 2:
                    a = stack.pop()
                    b = stack.pop()
                    stack.append(f"({a}{symbol}{b})")
                else:
                    stack.append(random.choice(variables))
    return stack[0] if stack else random.choice(variables)

def safe_eval(expr, C, F, W):
    try:
        expr = expr.replace('/0', '/1e-6')
        expr = expr.replace('log(', 'np.log(abs(')
        expr = expr.replace('sqrt(', 'np.sqrt(abs(')
        return eval(expr, {"C": C, "F": F, "W": W, "np": np, "math": math})
    except:
        return np.inf

def generate_individual():
    return [generate_gene() for _ in range(NUM_GENES)]

# ---------------- Structured Multi-Gene Combination ----------------
def individual_to_expression(individual):
    # Start: gene1
    start = gene_to_expression(individual[0])
    # Middle: combine gene2 and gene3 nonlinearly
    middle = f"({gene_to_expression(individual[1])} * sin({gene_to_expression(individual[2])}))"
    # End: some nonlinear transformation of last gene
    end = f"- sqrt({gene_to_expression(individual[-1])})"
    # Full structured expression
    return f"({start} + {middle} {end})"

def fitness(individual):
    expr = individual_to_expression(individual)
    C_vals, F_vals, W_vals = dataset[:, 0], dataset[:, 1], dataset[:, 2]
    y_true = dataset[:, 3]
    y_pred = np.array([safe_eval(expr, C_vals[i], F_vals[i], W_vals[i]) for i in range(len(C_vals))])
    return np.mean((y_true - y_pred) ** 2)

# ---------------- GEP OPERATORS ----------------
def mutate(individual):
    ind = [g[:] for g in individual]
    if random.random() < MUTATION_RATE:
        gene_idx = random.randint(0, NUM_GENES-1)
        gene = ind[gene_idx]
        idx = random.randint(0, len(gene)-1)
        if idx < HEAD_LENGTH:
            gene[idx] = random.choice(functions + variables)
        else:
            gene[idx] = random.choice(variables)
        ind[gene_idx] = gene
    return ind

def crossover(parent1, parent2):
    gene_idx = random.randint(0, NUM_GENES-1)
    pt = random.randint(1, len(parent1[gene_idx])-1)
    child1_gene = parent1[gene_idx][:pt] + parent2[gene_idx][pt:]
    child2_gene = parent2[gene_idx][:pt] + parent1[gene_idx][pt:]
    child1, child2 = parent1[:], parent2[:]
    child1[gene_idx], child2[gene_idx] = child1_gene, child2_gene
    return child1, child2

def select(population, k=3):
    return min(random.sample(population, k), key=fitness)

# ---------------- MAIN GEP LOOP ----------------
def main_gep():
    population = [generate_individual() for _ in range(POP_SIZE)]

    for gen in range(GENS):
        new_population = []
        while len(new_population) < POP_SIZE:
            parent1 = select(population)
            parent2 = select(population)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.extend([child1, child2])
        population = new_population[:POP_SIZE]

        best = min(population, key=fitness)
        best_expr = individual_to_expression(best)
        best_fit = fitness(best)
        print(f"Gen {gen}: Best expression: {best_expr}, MSE: {best_fit:.4f}")

    final_best = min(population, key=fitness)
    print("\nFinal Best Structured Expression (Compressive Strength in MPa):")
    print(individual_to_expression(final_best))
    print("Final MSE:", fitness(final_best))

if __name__ == "__main__":
    main_gep()
