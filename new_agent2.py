import numpy as np
import gymnasium as gym
import multiprocessing
from my_policy import policy_action
import os

# Hyperparameters
POPULATION_SIZE = 500        # Number of parameter samples per iteration
NUM_ELITE = 20               # Number of top-performing samples to retain
PARAM_DIM = 8 * 4 + 4        # Policy parameters (8 inputs * 4 actions + 4 biases)
NUM_EPISODES_EVAL = 100       # Episodes for fitness evaluation during evolution
FINAL_EPISODES_EVAL = 100    # Episodes for final validation
CHECKPOINT_FILE = "checkpoint_latest.npy"  # File to save the latest checkpoint
TARGET_AVG_REWARD = 350      # Target average reward to stop training
TOURNAMENT_SIZE = 10          # Number of individuals in each tournament
MUTATION_RATE = 0.05         # Probability of mutation per gene
MUTATION_STRENGTH = 0.2      # Strength of mutations

def evaluate_policy(params, num_episodes=NUM_EPISODES_EVAL):
    """Evaluate policy over specified number of episodes."""
    total_reward = 0.0
    for _ in range(num_episodes):
        env = gym.make("LunarLander-v3")
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action = policy_action(params, obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        total_reward += episode_reward
        env.close()
    return total_reward / num_episodes

def save_checkpoint(iteration, population, fitness, best_avg, best_params):
    """Save the current state of the training process."""
    checkpoint_data = {
        'iteration': iteration,
        'population': population,
        'fitness': fitness,
        'best_avg': best_avg,
        'best_params': best_params
    }
    np.save(CHECKPOINT_FILE, checkpoint_data)
    print(f"Checkpoint saved: {CHECKPOINT_FILE}")
    
    # Also save just the best parameters in a separate file
    # This can be loaded without allow_pickle=True
    if best_params is not None:
        np.save("checkpoint_best_params.npy", best_params)
        print(f"Best parameters saved separately for easy evaluation.")

def load_checkpoint(checkpoint_file):
    """Load the state of the training process from a checkpoint file."""
    if not os.path.exists(checkpoint_file):
        print(f"Checkpoint file {checkpoint_file} does not exist.")
        return None
    checkpoint_data = np.load(checkpoint_file, allow_pickle=True).item()
    print(f"Checkpoint loaded: {checkpoint_file}")
    return checkpoint_data

def tournament_selection(population, fitness, tournament_size):
    """Select an individual from the population using tournament selection."""
    selected_indices = np.random.choice(len(population), tournament_size, replace=False)
    selected_fitness = [fitness[i] for i in selected_indices]
    winner_index = selected_indices[np.argmax(selected_fitness)]
    return population[winner_index]

def crossover(parent1, parent2):
    """Perform crossover between two parents to produce an offspring."""
    # Choose a random crossover point
    crossover_point = np.random.randint(1, PARAM_DIM-1)
    child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    return child

def mutate(individual):
    """Mutate an individual by adding Gaussian noise to each gene with probability MUTATION_RATE."""
    mutated = individual.copy()
    mask = np.random.random(PARAM_DIM) < MUTATION_RATE
    noise = np.random.normal(0, MUTATION_STRENGTH, PARAM_DIM)
    mutated = mutated + mask * noise
    return mutated

def main():
    # Check for existing checkpoint
    checkpoint_data = load_checkpoint(CHECKPOINT_FILE)
    
    if checkpoint_data is not None:
        iteration = checkpoint_data['iteration']
        population = checkpoint_data['population']
        fitness = checkpoint_data['fitness']
        best_avg = checkpoint_data['best_avg']
        best_params = checkpoint_data['best_params']
        print(f"Resuming from iteration {iteration}")
    else:
        # Initialize population randomly
        population = np.random.normal(0, 1, (POPULATION_SIZE, PARAM_DIM))
        fitness = np.zeros(POPULATION_SIZE)
        best_avg = -np.inf
        best_params = None
        iteration = 0

    # Start multiprocessing pool
    with multiprocessing.Pool() as pool:
        while best_avg < TARGET_AVG_REWARD:
            # Evaluate fitness of each individual in the population
            fitness = pool.map(evaluate_policy, population)
            
            # Calculate population average
            population_avg = np.mean(fitness)
            
            # Find the best individual in this generation
            best_index = np.argmax(fitness)
            best_fitness = fitness[best_index]
            
            # Update best parameters if we found a better one
            if best_fitness > best_avg:
                best_avg = best_fitness
                best_params = population[best_index].copy()
                # Re-evaluate best params with more episodes for better estimate
                confirm_fitness = evaluate_policy(best_params, num_episodes=10)
                print(f"\nIteration {iteration+1}: New best candidate found!")
                print(f"Initial fitness: {best_fitness:.2f}, Confirmed fitness (10 episodes): {confirm_fitness:.2f}")
                if confirm_fitness > 200:  # Only save if it's reasonably good
                    np.save("best_policy.npy", best_params)
                    print(f"Saved new best policy with confirmed fitness: {confirm_fitness:.2f}")
            
            # Print statistics
            print(f"Generation {iteration+1}: Population Avg: {population_avg:.2f}, Best Fitness: {best_fitness:.2f}, Overall Best: {best_avg:.2f}")
            
            # Create new population with elitism (keep best individuals)
            elite_indices = np.argsort(fitness)[::-1][:NUM_ELITE]
            new_population = [population[i].copy() for i in elite_indices]
            
            # Fill the rest of the population with offspring
            while len(new_population) < POPULATION_SIZE:
                # Select parents using tournament selection
                parent1 = tournament_selection(population, fitness, TOURNAMENT_SIZE)
                parent2 = tournament_selection(population, fitness, TOURNAMENT_SIZE)
                
                # Create offspring through crossover and mutation
                child = crossover(parent1, parent2)
                child = mutate(child)
                new_population.append(child)
            
            population = np.array(new_population)
            
            # Save checkpoint after each iteration
            save_checkpoint(iteration + 1, population, fitness, best_avg, best_params)
            
            iteration += 1

    # Final evaluation with more episodes
    if best_params is not None:
        final_avg = evaluate_policy(best_params, num_episodes=FINAL_EPISODES_EVAL)
        print(f"\nFinal best average reward over {FINAL_EPISODES_EVAL} episodes: {final_avg:.2f}")
        np.save("best_policy_final.npy", best_params)
        print("Saved final best policy.")

if __name__== "__main__":
    main()
