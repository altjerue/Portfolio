import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import time

print(f"JAX version: {jax.__version__}")

# Test basic operations
x = jnp.array([1, 2, 3])
print(f"JAX array: {x}")
print(f"Type: {type(x)}")


# MARK: JAX
"""
WHY IMMUTABILITY?
- Makes code parallelizable (no race conditions)
- Enable aggressive compiler optimizations
- Allows functional programming patterns
- Critical for reproducible scientific computing
"""

# NumPy: mutable arrays (can cause bugs in parallel code)
x_numpy = np.array([1, 2, 3, 4, 5])
x_numpy[0] = 99  # This works
print(f"Numpy modified: {x_numpy}")

# JAX: immutable arrays (safe for parallel executions)
x_jax = jnp.array([1, 2, 3, 4, 5])

# This would ERROR:
# x_jax[0] = 99 # TypeError!

# Correct JAX way:
x_jax = x_jax.at[0].set(99)
print(f"JAX modified: {x_jax}")

# Common patterns you'll use:
x = jnp.array([1, 2, 3, 4, 5])

# Update single element
x = x.at[2].set(100)  # x[2] = 100

# Update multiple elements
x = x.at[1:4].set(0)  # x[1:4] = 0

# Increment (commmon in simulations)
x = x.at[0].add(10)  # x[0] += 10

# Conditional update
mask = x > 50
x = x.at[mask].set(-1)


# EXERCISE 1: Rewriting this NumPy code in JAX
def update_temperatures_numpy(temps):
    """NumPy: Update temperatures that eceed threshold"""
    temps = temps.copy()  # Avoid modifying original
    hot = temps > 30
    temps[hot] = 30  # Cap at 30
    temps[temps < 0] = 0  # Floor at 0
    return temps


def update_temperatures_jax(temps):
    """JAX version of update_temperatures_numpy"""
    # TODO: Implement using .at[] syntax
    # temps = temps.copy() # Q: do I need to make a copy for JAX?
    hot = temps > 30
    temps = temps.at[hot].set(30)
    temps = temps.at[temps < 0].set(0)
    return temps


# Test your implementation
temps_np = np.array([25, 35, -5, 28, 40, 15])
temps_jax = jnp.array([25, 35, -5, 28, 40, 15])

print("Numpy results: ", update_temperatures_numpy(temps_np))
print("JAX results:   ", update_temperatures_jax(temps_jax))
print("=" * 60)

"""
CHECKPOINT 1: Can you explain why immutability matters for parallel computing?

Answer: Immutable arrays allow to prevent race conditions when parallelizing
        computations. This also ensures that the compilation will use higher
        optimization options.
"""

# ==============================================================================

# MARK: Random Numbers
"""
WHY DIFFERENT RANDOMNESS?
- NumPy uses global state (not reproducible in parallel)
- JAX uses explicit keys (fully reproducible)
- Allows splitting random streams for parallel execution
"""

from jax import random

# JAX requires explicit random key
key = random.PRNGKey(42)
print(f"Random key: {key}")

# Generate random numbers
random_number = random.uniform(key, shape=())
print(f"Random number: {random_number}")

# IMPORTANT: Using same key gives same result!
key = random.PRNGKey(42)
print(f"Same key, same result: {random.uniform(key, shape=())}")

# To get different numbers, SPLIT the key
key = random.PRNGKey(42)
key, subkey = random.split(key)
print(f"First random: {random.uniform(subkey, shape=())}")

key, subkey = random.split(key)
print(f"Second random: {random.uniform(subkey, shape=())}")

# For multiple random values, split multiple times
key = random.PRNGKey(42)
keys = random.split(key, num=5)
print(f"Shape of keys: {keys.shape}")

# Generate 5 independent random numbers
random_numbers = jnp.array([random.uniform(k) for k in keys])
print(f"5 random numbers: {random_numbers}")

# Better: Generate array directly
key = random.PRNGKey(42)
random_array = random.uniform(key, shape=(5,))
print(f"Random array: {random_array}")

# Common distributions for simulations
key = random.PRNGKey(42)
key, *subkeys = random.split(key, 5)

normal = random.normal(subkeys[0], shape=(100,))  # Gaussian
uniform = random.uniform(subkeys[1], shape=(100,))  # Uniform [0,1)
exponential = random.exponential(subkeys[2], shape=(100,))  # Exponential
bernoulli = random.bernoulli(subkeys[3], p=0.3, shape=(100,))  # 0/1


# EXERCISE 2: Monte Carlo Pi Estimation (NumPy style)
def monte_carlo_pi_numpy(n_samples):
    """Estimate pi using Monte Carlo - NumPy version"""
    x = np.random.uniform(0, 1, n_samples)
    y = np.random.uniform(0, 1, n_samples)
    inside = (x**2 + y**2) <= 1
    return 4 * np.sum(inside) / n_samples


# EXERCISE 2: JAX Version
def monte_carlo_pi_jax(key, n_samples):
    """Estimate pi using Monte Carlo - JAX version"""
    # TODO: Implement using JAX random
    # Need to handle random key properly
    x = random.uniform(key, shape=(n_samples,))
    y = random.uniform(key, shape=(n_samples,))
    inside = (x**2 + y**2) <= 1
    return 4 * jnp.sum(inside) / n_samples


# Test
pi_estimate = monte_carlo_pi_numpy(1_000_000)
print(f"NumPy pi estimate: {pi_estimate:.6f}")
print(f"Error: {abs(pi_estimate - np.pi):.6f}")

key = random.PRNGKey(42)
pi_estimate = monte_carlo_pi_jax(key, 1_000_000)
print(f"JAX pi estimate: {pi_estimate:.6f}")
print(f"Error: {abs(pi_estimate - np.pi):.6f}")
print("=" * 60)

# ==============================================================================

# MARK: JIT
"""
JIT = Just-In-Time compilation
- Compiles Python/JAX code to optimized machine code
- First call is slow (compilation)
- Subsequent calls are FAST (10-100x speedup)
- This is why Alaska Airlines uses JAX
"""

from jax import jit


# WITHOUT JIT (slow)
def slow_computation(x):
    """Complex computation without JIT"""

    for _ in range(10):
        x = jnp.sin(x) + jnp.cos(x)
        x = jnp.sqrt(jnp.abs(x))
    return jnp.sum(x)


# WITH JIT (fast)
@jit
def fast_computation(x):
    """Same computation with JIT"""
    for _ in range(10):
        x = jnp.sin(x) + jnp.cos(x)
        x = jnp.sqrt(jnp.abs(x))
    return jnp.sum(x)


# Benchmark
x = jnp.ones(1_000_000)

# Slow version
start = time.time()
result_slow = slow_computation(x)
slow_time = time.time() - start

# Fast version - FIRST CALL (includes compilation)
start = time.time()
result_fast = fast_computation(x)
first_call_time = time.time() - start

# Fast version - SECOND CALL (no compilation)
start = time.time()
result_fast = fast_computation(x)
fast_time = time.time() - start

print(f"Slow version: {slow_time:.4f} seconds")
print(f"Fast version (first call): {first_call_time:.4f} seconds")
print(f"Fast version (subsequent): {fast_time:.4f} seconds")
print(f"Speedup: {slow_time / fast_time:.1f}x")
print(f"Results match: {jnp.allclose(result_slow, result_fast)}")
print("=" * 60)

# ==============================================================================

# MARK: JIT Rules

"""
JIT RULES
1. Function must be "pure" (same input -> same output)
2. No side effects (no printing, file I/O during computation)
3. Control flow based on traced values can be tricky
4. Array shapes should be static when possible
"""


# GOOD: Pure function
@jit
def good_function(x, y):
    return x + y


# BAD: has side effects (print)
@jit
def bad_function_print(x):
    print(f"Value: {x}")  # This won't work as expected!
    return x + 1


# BAD: Control flow based on value
@jit
def bad_function_control(x):
    if x > 0:  # This is tricky with JIT!
        return x
    else:
        return -x


# GOOD: Use jnp.where for conditional logic
@jit
def good_function_control(x):
    return jnp.where(x > 0, x, -x)


# BAD: Dynamic shapes
@jit
def bad_function_dynamic(x, threshold):
    return x[x > threshold]  # Output size depends on values!


# GOOD: Static shapes
@jit
def good_function_static(x, threshold):
    mask = x > threshold
    # Return both mask and values, or use padding
    return jnp.where(mask, x, 0)


# EXERCISE 3: Fix this function to work with JIT
def compute_statistics_bad(data):
    """Compute statistics on data."""
    print(f"Processing {len(data)} samples")  # Issue 1: print

    # Issue 2: dynamic sclicing
    positive = data[data > 0]
    negative = data[data <= 0]

    # Issue 3: if statement on traced value
    if len(positive) > len(negative):
        return jnp.mean(positive)
    else:
        return jnp.mean(negative)


# FIXED VERSION:
@jit
def compute_statistics_fixed(data):
    """JIT-compatible version."""
    mask = data > 0
    positive = jnp.where(mask, data, 0)
    negative = jnp.where(~mask, data, 0)

    return jnp.where(mask, jnp.mean(positive), jnp.mean(negative))


# Test
data = jnp.array([1, -2, 3, -4, 5, 6, -1])
result = compute_statistics_fixed(data)
print(f"Result: {result}")
