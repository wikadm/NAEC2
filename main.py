####Neural and Evolutionary Computation (NEC) - Assignment 2
####Wiktor Samulski

import numpy as np
import random
import os
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict


RANDOM_SEED = 42
#add seeds for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

#########JSSP instance and Chromosome Classes
class JSSPInstance:
#####Represents a Job Shop Scheduling Problem instance

    def __init__(self, jobs: List[List[Tuple[int, int]]], name: str = ""):
        self.jobs = jobs
        self.name = name
        self.num_jobs = len(jobs)
##Calculate max mchine ID to find out total machines
        self.num_machines = max(max(op[0] for op in job) for job in jobs) + 1
        self.num_operations = sum(len(job) for job in jobs)


class Chromosome:
###Represents a solution using operation-based representation.
###The chromosome is a permutation of job indices where each job appears as many times as it has operations.

    def __init__(self, genes: List[int], instance: JSSPInstance):
        self.genes = genes
        self.instance = instance
        self._fitness = None
        self._schedule = None

    @classmethod
    def create_random(cls, instance: JSSPInstance) -> 'Chromosome':
#random valid permutation of job ID
        genes = []
        for job_id, job in enumerate(instance.jobs):
##add job_id n times
##n is number of operations of that job
            genes.extend([job_id] * len(job))
        random.shuffle(genes)
        return cls(genes, instance)

    @property
    def fitness(self) -> int:
##score of fitness by Lazy Evaluation and Property Decorator
        if self._fitness is None:
            self._fitness, self._schedule = self._calculate_makespan()
        return self._fitness

    @property
    def schedule(self) -> Dict:
        if self._schedule is None:
            self._fitness, self._schedule = self._calculate_makespan()
        return self._schedule

    def _calculate_makespan(self) -> Tuple[int, Dict]:
##decodes chromosome into schedule
        job_op_index = [0] * self.instance.num_jobs
        job_end_times = [0] * self.instance.num_jobs
        machine_end_times = [0] * self.instance.num_machines
        schedule = {m: [] for m in range(self.instance.num_machines)}

        for job_id in self.genes:
            op_idx = job_op_index[job_id]
            machine, duration = self.instance.jobs[job_id][op_idx]
            start_time = max(job_end_times[job_id], machine_end_times[machine])
            end_time = start_time + duration

            job_end_times[job_id] = end_time
            machine_end_times[machine] = end_time
            job_op_index[job_id] += 1

            schedule[machine].append({
                'job': job_id, 'operation': op_idx,
                'start': start_time, 'end': end_time, 'duration': duration
            })

        return max(job_end_times), schedule

    def is_valid(self) -> bool:
##validation of the chromosome
##if every job appears as many times as it has operations
        job_counts = {}
        for gene in self.genes:
            job_counts[gene] = job_counts.get(gene, 0) + 1
        for job_id, job in enumerate(self.instance.jobs):
            if job_counts.get(job_id, 0) != len(job):
                return False
        return True

    def copy(self) -> 'Chromosome':
        return Chromosome(self.genes.copy(), self.instance)

######Selection Methods
##randomly picks k and return of best one
def tournament_selection(population: List[Chromosome], tournament_size: int = 3) -> Chromosome:
        tournament = random.sample(population, min(tournament_size, len(population)))
        return min(tournament, key=lambda c: c.fitness)


def roulette_wheel_selection(population: List[Chromosome]) -> Chromosome:
##select based on probability proportional to fitness
    max_fitness = max(c.fitness for c in population)
    inverted_fitness = [max_fitness - c.fitness + 1 for c in population]
    total = sum(inverted_fitness)
    if total == 0:
        return random.choice(population)
    probs = [f / total for f in inverted_fitness]
    return random.choices(population, weights=probs, k=1)[0]


def rank_selection(population: List[Chromosome]) -> Chromosome:
##rank selection
    sorted_pop = sorted(population, key=lambda c: c.fitness)
    n = len(sorted_pop)
    ranks = list(range(n, 0, -1))
    total = sum(ranks)
    probs = [r / total for r in ranks]
    return random.choices(sorted_pop, weights=probs, k=1)[0]

###Crossover methods
#copy form parent 1, fills remaining spots of genes in Parent2
def order_crossover_ox(parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
    size = len(parent1.genes)
    point1, point2 = sorted(random.sample(range(size), 2))

    def create_offspring(p1, p2):
        offspring = [None] * size
        for i in range(point1, point2 + 1):
            offspring[i] = p1.genes[i]

        gene_counts = {}
        for gene in p2.genes:
            gene_counts[gene] = gene_counts.get(gene, 0) + 1
        for gene in offspring:
            if gene is not None:
                gene_counts[gene] -= 1

        p2_idx = 0
        for i in list(range(point2 + 1, size)) + list(range(0, point1)):
            while gene_counts.get(p2.genes[p2_idx], 0) <= 0:
                p2_idx += 1
            offspring[i] = p2.genes[p2_idx]
            gene_counts[p2.genes[p2_idx]] -= 1
            p2_idx += 1
        return Chromosome(offspring, p1.instance)

    return create_offspring(parent1, parent2), create_offspring(parent2, parent1)


def pmx_crossover(parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
##maps genes between the cut points
    size = len(parent1.genes)
    point1, point2 = sorted(random.sample(range(size), 2))

    def create_offspring(p1, p2):
        offspring = [None] * size
        for i in range(point1, point2 + 1):
            offspring[i] = p1.genes[i]

        gene_counts = {}
        for gene in p2.genes:
            gene_counts[gene] = gene_counts.get(gene, 0) + 1
        for gene in offspring:
            if gene is not None:
                gene_counts[gene] -= 1

        p2_idx = 0
        for i in range(size):
            if offspring[i] is None:
                while gene_counts.get(p2.genes[p2_idx], 0) <= 0:
                    p2_idx += 1
                offspring[i] = p2.genes[p2_idx]
                gene_counts[p2.genes[p2_idx]] -= 1
                p2_idx += 1
        return Chromosome(offspring, p1.instance)

    return create_offspring(parent1, parent2), create_offspring(parent2, parent1)
###mutation methods
#swap two genes
def swap_mutation(chromosome: Chromosome, rate: float = 0.1) -> Chromosome:
    if random.random() > rate:
        return chromosome.copy()
    genes = chromosome.genes.copy()
    i, j = random.sample(range(len(genes)), 2)
    genes[i], genes[j] = genes[j], genes[i]
    return Chromosome(genes, chromosome.instance)


def insertion_mutation(chromosome: Chromosome, rate: float = 0.1) -> Chromosome:
#insert gene into a new random position
    if random.random() > rate:
        return chromosome.copy()
    genes = chromosome.genes.copy()
    i = random.randint(0, len(genes) - 1)
    gene = genes.pop(i)
    j = random.randint(0, len(genes))
    genes.insert(j, gene)
    return Chromosome(genes, chromosome.instance)
####genetic algorithm
class GeneticAlgorithm:
    
    SELECTION = {'tournament': tournament_selection, 'roulette': roulette_wheel_selection, 'rank': rank_selection}
    CROSSOVER = {'ox': order_crossover_ox, 'pmx': pmx_crossover}
    MUTATION = {'swap': swap_mutation, 'insertion': insertion_mutation}
    
    def __init__(self, instance: JSSPInstance, pop_size: int = 100,
                 selection: str = 'tournament', crossover: str = 'ox', mutation: str = 'swap',
                 crossover_rate: float = 0.8, mutation_rate: float = 0.1,
                 elitism: int = 2, tournament_size: int = 3):
        
        self.instance = instance
        self.pop_size = pop_size
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.tournament_size = tournament_size
        
        self._select = self.SELECTION[selection]
        self._crossover = self.CROSSOVER[crossover]
        self._mutate = self.MUTATION[mutation]
        
        self.history = {'generation': [], 'best': [], 'avg': [], 'worst': []}
    
    def run(self, max_gen: int = 300, stagnation: int = 50, verbose: bool = True) -> Tuple[Chromosome, Dict]:
        population = [Chromosome.create_random(self.instance) for _ in range(self.pop_size)]
        best_ever = min(population, key=lambda c: c.fitness)
        stag_count = 0
        
        for gen in range(max_gen):
            # Create new population
            new_pop = []
            sorted_pop = sorted(population, key=lambda c: c.fitness)
            new_pop.extend([c.copy() for c in sorted_pop[:self.elitism]])
            # reprodiction loop 
            while len(new_pop) < self.pop_size:
                # Selection
                if self.selection == 'tournament':
                    p1 = self._select(population, self.tournament_size)
                    p2 = self._select(population, self.tournament_size)
                else:
                    p1, p2 = self._select(population), self._select(population)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    c1, c2 = self._crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                
                # Mutation
                c1 = self._mutate(c1, self.mutation_rate)
                c2 = self._mutate(c2, self.mutation_rate)
                
                if c1.is_valid():
                    new_pop.append(c1)
                if len(new_pop) < self.pop_size and c2.is_valid():
                    new_pop.append(c2)
            
            population = new_pop[:self.pop_size]
            
            # Statistics
            fitnesses = [c.fitness for c in population]
            best_gen = min(population, key=lambda c: c.fitness)
            
            self.history['generation'].append(gen)
            self.history['best'].append(min(fitnesses))
            self.history['avg'].append(np.mean(fitnesses))
            self.history['worst'].append(max(fitnesses))
            
            if best_gen.fitness < best_ever.fitness:
                best_ever = best_gen.copy()
                stag_count = 0
            else:
                stag_count += 1
            
            if verbose and gen % 20 == 0:
                print(f"Gen {gen:4d}: Best={min(fitnesses)}, Avg={np.mean(fitnesses):.1f}, Best Ever={best_ever.fitness}")
            
            if stag_count >= stagnation:
                if verbose:
                    print(f"Stopping at generation {gen} (stagnation)")
                break
        
        if verbose:
            print(f"\nFinal Best Makespan: {best_ever.fitness}")
        
        return best_ever, self.history
###visualization functions
def plot_evolution(history: Dict, title: str = "GA Evolution", save_path: str = None):
    plt.figure(figsize=(10, 5))
    plt.plot(history['generation'], history['best'], label='Best', linewidth=2, color='green')
    plt.plot(history['generation'], history['avg'], label='Average', linewidth=1.5, color='blue', alpha=0.7)
    plt.plot(history['generation'], history['worst'], label='Worst', linewidth=1, color='red', alpha=0.5)
    plt.xlabel('Generation')
    plt.ylabel('Makespan')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_gantt(schedule: Dict, instance: JSSPInstance, title: str = "Schedule", save_path: str = None):
#draw a gantt chart
    colors = plt.cm.Set3(np.linspace(0, 1, instance.num_jobs))
    fig, ax = plt.subplots(figsize=(12, max(4, instance.num_machines * 0.6)))

    for machine_id, tasks in schedule.items():
        for task in tasks:
            ax.barh(machine_id, task['duration'], left=task['start'], height=0.6,
                   color=colors[task['job']], edgecolor='black', linewidth=0.5)
            ax.text(task['start'] + task['duration']/2, machine_id, f"J{task['job']}",
                   ha='center', va='center', fontsize=8, fontweight='bold')

    ax.set_yticks(range(instance.num_machines))
    ax.set_yticklabels([f'M{i}' for i in range(instance.num_machines)])
    ax.set_xlabel('Time')
    ax.set_title(title)
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_comparison(results: List[Dict], title: str = "Configuration Comparison", save_path: str = None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for r in results:
        label = f"{r['selection']}/{r['crossover']}/{r['mutation']}"
        axes[0].plot(r['history']['generation'], r['history']['best'], label=label, linewidth=1.5)
    axes[0].set_xlabel('Generation')
    axes[0].set_ylabel('Best Makespan')
    axes[0].set_title('Convergence')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    labels = [f"C{i+1}" for i in range(len(results))]
    makespans = [r['best_fitness'] for r in results]
    bars = axes[1].bar(labels, makespans, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(results))))
    axes[1].set_ylabel('Best Makespan')
    axes[1].set_title('Final Results')
    for bar, val in zip(bars, makespans):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(val), ha='center', fontsize=10)
    axes[1].grid(True, axis='y', alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()

#####benchmark instances
def create_small_instance():
#3 jobs, 3 machines. Simple test case.
    return JSSPInstance([
        [(0, 3), (1, 2), (2, 2)],
        [(0, 2), (2, 1), (1, 4)],
        [(1, 4), (2, 3)]
    ], "small_3x3")


def create_medium_instance():
##LA16 10 jobs x 10 machines
    return JSSPInstance([
        [(1, 21), (6, 71), (9, 16), (8, 52), (7, 26), (2, 34), (0, 53), (4, 21), (3, 55), (5, 95)],
        [(4, 55), (2, 31), (5, 98), (9, 79), (0, 12), (7, 66), (1, 42), (8, 77), (6, 77), (3, 39)],
        [(3, 34), (2, 64), (8, 62), (1, 19), (4, 92), (9, 79), (7, 43), (6, 54), (0, 83), (5, 37)],
        [(1, 87), (3, 69), (2, 87), (7, 38), (8, 24), (9, 83), (6, 41), (0, 93), (5, 77), (4, 60)],
        [(2, 98), (0, 44), (5, 25), (6, 75), (7, 43), (1, 49), (4, 96), (9, 77), (3, 17), (8, 79)],
        [(2, 35), (3, 76), (5, 28), (9, 10), (4, 61), (6, 9), (0, 95), (8, 35), (1, 7), (7, 95)],
        [(3, 16), (2, 59), (0, 46), (1, 91), (9, 43), (8, 50), (6, 52), (5, 59), (4, 28), (7, 27)],
        [(1, 45), (0, 87), (3, 41), (4, 20), (6, 54), (9, 43), (8, 14), (5, 9), (2, 39), (7, 71)],
        [(4, 33), (3, 37), (8, 66), (5, 33), (2, 26), (7, 8), (1, 28), (6, 89), (9, 42), (0, 78)],
        [(8, 69), (9, 81), (2, 94), (4, 96), (6, 27), (0, 69), (7, 45), (3, 78), (1, 74), (5, 84)]
    ], "la16_10x10")


def create_large_instance():
#LA36 15 jobs x 15 machines
    return JSSPInstance([
        [(4, 21), (3, 34), (14, 95), (0, 53), (8, 55), (6, 16), (10, 52), (9, 21), (2, 26), (12, 71), (5, 39), (11, 98), (1, 42), (7, 31), (13, 12)],
        [(1, 77), (4, 66), (2, 79), (13, 55), (3, 77), (14, 69), (0, 8), (12, 83), (7, 34), (10, 64), (8, 19), (5, 37), (6, 54), (9, 43), (11, 79)],
        [(8, 92), (7, 62), (2, 69), (10, 77), (9, 87), (4, 87), (1, 93), (6, 38), (5, 60), (14, 41), (12, 24), (0, 83), (3, 17), (13, 49), (11, 25)],
        [(0, 44), (2, 98), (14, 77), (5, 79), (3, 43), (10, 75), (1, 96), (12, 28), (7, 7), (8, 95), (4, 35), (6, 76), (9, 51), (11, 10), (13, 61)],
        [(4, 9), (6, 85), (12, 59), (11, 13), (5, 85), (9, 89), (10, 45), (2, 33), (1, 81), (0, 95), (7, 71), (3, 99), (14, 9), (8, 52), (13, 85)],
        [(7, 98), (1, 22), (6, 43), (5, 14), (3, 6), (9, 22), (2, 61), (12, 26), (10, 69), (4, 21), (14, 49), (11, 72), (0, 53), (8, 84), (13, 2)],
        [(13, 52), (6, 95), (11, 48), (10, 72), (12, 47), (1, 65), (8, 6), (4, 25), (3, 46), (9, 37), (0, 61), (7, 13), (14, 32), (2, 21), (5, 32)],
        [(5, 89), (14, 30), (1, 55), (6, 31), (11, 86), (2, 46), (13, 74), (0, 32), (10, 88), (4, 19), (3, 48), (9, 36), (12, 79), (8, 76), (7, 69)],
        [(11, 76), (1, 51), (12, 85), (6, 11), (0, 40), (10, 89), (4, 26), (9, 74), (14, 85), (7, 13), (8, 61), (2, 7), (5, 64), (3, 76), (13, 47)],
        [(6, 52), (4, 90), (14, 45), (12, 6), (7, 23), (2, 95), (9, 82), (5, 6), (10, 84), (0, 14), (11, 35), (1, 59), (3, 62), (13, 47), (8, 17)],
        [(9, 75), (8, 98), (3, 35), (7, 43), (5, 37), (4, 85), (0, 72), (14, 54), (13, 46), (10, 59), (2, 28), (1, 62), (11, 58), (12, 27), (6, 9)],
        [(4, 76), (6, 27), (7, 87), (14, 99), (12, 96), (13, 27), (5, 80), (2, 43), (11, 77), (3, 62), (0, 45), (1, 3), (8, 24), (9, 38), (10, 82)],
        [(2, 44), (8, 5), (3, 15), (12, 35), (9, 62), (13, 79), (4, 92), (7, 3), (6, 59), (1, 73), (14, 61), (10, 32), (5, 37), (11, 55), (0, 51)],
        [(12, 84), (0, 25), (14, 59), (1, 19), (10, 80), (5, 44), (4, 16), (3, 55), (2, 29), (9, 69), (7, 73), (8, 37), (13, 29), (11, 47), (6, 84)],
        [(1, 48), (2, 51), (14, 17), (7, 46), (13, 58), (10, 70), (0, 98), (11, 63), (3, 86), (9, 10), (6, 62), (5, 75), (12, 76), (8, 90), (4, 23)]
    ], "la36_15x15")

##########main
def run_experiments():
    ##Run experiments with different GA configurations.

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Create output directory
    os.makedirs("results", exist_ok=True)

    # Test instances
    instances = [create_small_instance(), create_medium_instance(), create_large_instance()]

    # GA configurations
    configs = [
        {'selection': 'tournament', 'crossover': 'ox', 'mutation': 'swap', 'mutation_rate': 0.15},
        {'selection': 'tournament', 'crossover': 'pmx', 'mutation': 'insertion', 'mutation_rate': 0.15},
        {'selection': 'roulette', 'crossover': 'ox', 'mutation': 'swap', 'mutation_rate': 0.10},
        {'selection': 'rank', 'crossover': 'ox', 'mutation': 'insertion', 'mutation_rate': 0.12},
        {'selection': 'tournament', 'crossover': 'ox', 'mutation': 'insertion', 'mutation_rate': 0.08},
        {'selection': 'rank', 'crossover': 'pmx', 'mutation': 'swap', 'mutation_rate': 0.20},
    ]

    all_results = {}

    for instance in instances:
        print(f"\n{'='*60}")
        print(f"Instance: {instance.name}")
        print(f"Jobs: {instance.num_jobs}, Machines: {instance.num_machines}")
        print(f"{'='*60}")

        results = []

        for i, cfg in enumerate(configs):
            print(f"\nConfig {i+1}: {cfg['selection']}/{cfg['crossover']}/{cfg['mutation']}")

            random.seed(RANDOM_SEED)
            np.random.seed(RANDOM_SEED)

            ga = GeneticAlgorithm(
                instance, pop_size=100,
                selection=cfg['selection'], crossover=cfg['crossover'], mutation=cfg['mutation'],
                crossover_rate=0.85, mutation_rate=cfg['mutation_rate'],
                elitism=2, tournament_size=5
            )

            best, history = ga.run(max_gen=300, stagnation=50, verbose=False)

            results.append({
                'config': i+1,
                'selection': cfg['selection'],
                'crossover': cfg['crossover'],
                'mutation': cfg['mutation'],
                'best_fitness': best.fitness,
                'best_chromosome': best,
                'history': history,
                'generations': len(history['generation'])
            })

            print(f"  Best Makespan: {best.fitness}, Generations: {len(history['generation'])}")

        # Find best result
        best_result = min(results, key=lambda r: r['best_fitness'])

        # Generate plots
        plot_evolution(
            best_result['history'],
            f"GA Evolution - {instance.name}",
            f"results/{instance.name}_evolution.png"
        )

        plot_gantt(
            best_result['best_chromosome'].schedule,
            instance,
            f"Best Schedule - {instance.name} (Makespan: {best_result['best_fitness']})",
            f"results/{instance.name}_gantt.png"
        )

        plot_comparison(
            results,
            f"Configuration Comparison - {instance.name}",
            f"results/{instance.name}_comparison.png"
        )

        all_results[instance.name] = results

        # Print summary table
        print(f"\n{'Config':<10} {'Selection':<12} {'Crossover':<10} {'Mutation':<12} {'Makespan':<10} {'Gens':<8}")
        print("-" * 62)
        for r in results:
            print(f"{r['config']:<10} {r['selection']:<12} {r['crossover']:<10} {r['mutation']:<12} {r['best_fitness']:<10} {r['generations']:<8}")
        print(f"\nBest: {best_result['best_fitness']} (Config {best_result['config']})")

    print(f"\n{'='*60}")
    print("Results saved to 'results/' directory")
    print(f"{'='*60}")

    return all_results


if __name__ == "__main__":
    results = run_experiments()

