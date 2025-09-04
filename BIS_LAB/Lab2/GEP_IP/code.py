import random
import numpy as np
from PIL import Image, ImageEnhance
from skimage.metrics import structural_similarity as ssim
from skimage.filters import sobel
from skimage import img_as_float
import math

# ----- Load Image -----
original = Image.open("input.jpg").convert("RGB")
original_np = np.array(original)

try:
    target = Image.open("target.jpg").convert("RGB")
    target_np = np.array(target)
    USE_REFERENCE = True
except:
    USE_REFERENCE = False

# ----- Function Set -----
def protected_div(x, y):
    try:
        return x / y if abs(y) > 1e-6 else 1.0
    except:
        return 1.0

FUNCTIONS = [
    (lambda x: x, 'x'),
    (np.sin, 'sin'),
    (np.cos, 'cos'),
    (np.tan, 'tan'),
    (np.exp, 'exp'),
    (np.log1p, 'log1p'),
    (lambda x: x**2, 'square'),
    (np.sqrt, 'sqrt'),
    (np.abs, 'abs'),
]

# ----- Generate Random Expression Trees -----
def random_expr(depth=2):
    if depth == 0 or random.random() < 0.3:
        return 'x'
    func = random.choice(FUNCTIONS)[1]
    sub = random_expr(depth - 1)
    return f"{func}({sub})"

def evaluate_expr(expr_str, x_val):
    try:
        x = x_val
        return eval(expr_str, {"x": x, "sin": np.sin, "cos": np.cos, "tan": np.tan,
                               "exp": np.exp, "log1p": np.log1p, "sqrt": np.sqrt,
                               "abs": np.abs, "square": lambda x: x**2})
    except Exception as e:
        return 1.0

# ----- Individual is 3 Expressions -----
def init_population(size):
    return [[random_expr(2) for _ in range(3)] for _ in range(size)]

# ----- Apply Parameters -----
def apply_adjustments_from_expr(exprs):
    x = np.mean(original_np) / 255.0
    b = np.clip(evaluate_expr(exprs[0], x), 0.5, 2.0)
    c = np.clip(evaluate_expr(exprs[1], x), 0.5, 2.0)
    s = np.clip(evaluate_expr(exprs[2], x), 0.5, 2.0)

    img = original.copy()
    img = ImageEnhance.Brightness(img).enhance(b)
    img = ImageEnhance.Contrast(img).enhance(c)
    img = ImageEnhance.Sharpness(img).enhance(s)
    return img, [b, c, s]

# ----- Fitness -----
def fitness(ind):
    img, _ = apply_adjustments_from_expr(ind)
    img_np = np.array(img)

    if USE_REFERENCE:
        return ssim(target_np, img_np, channel_axis=2)
    else:
        gray = img.convert("L")
        gray_np = img_as_float(np.array(gray))
        entropy = -np.sum(gray_np * np.log2(gray_np + 1e-10))
        edge_strength = np.mean(sobel(gray_np))
        return entropy + edge_strength

# ----- Selection, Crossover, Mutation -----
def select(pop, fitnesses):
    i, j = random.sample(range(len(pop)), 2)
    return pop[i] if fitnesses[i] > fitnesses[j] else pop[j]

def crossover(p1, p2):
    child = []
    for a, b in zip(p1, p2):
        if random.random() < 0.5:
            child.append(a)
        else:
            child.append(b)
    return child

def mutate(expr):
    expr_list = expr.split()
    if random.random() < 0.3:
        return random_expr(2)
    return expr

def mutate_ind(ind):
    return [mutate(expr) if random.random() < 0.3 else expr for expr in ind]

# ----- Run GEP -----
def run_gep():
    POP_SIZE = 20
    N_GEN = 15
    ELITE = 2

    pop = init_population(POP_SIZE)

    for gen in range(N_GEN):
        fitnesses = [fitness(ind) for ind in pop]
        ranked = sorted(zip(pop, fitnesses), key=lambda x: x[1], reverse=True)
        best_ind, best_fit = ranked[0]

        print(f"Gen {gen}: Best fitness = {best_fit:.4f}, Exprs = {best_ind}")

        new_pop = [ind.copy() for ind, _ in ranked[:ELITE]]

        while len(new_pop) < POP_SIZE:
            p1, p2 = select(pop, fitnesses), select(pop, fitnesses)
            child = crossover(p1, p2)
            child = mutate_ind(child)
            new_pop.append(child)

        pop = new_pop

    best_img, best_params = apply_adjustments_from_expr(best_ind)
    best_img.save("gep_optimized.jpg")
    print("Best expressions found:", best_ind)
    print("Parameter values from expressions:", best_params)
    display(best_img)  # for Jupyter

# Run it
run_gep()
