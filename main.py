####Neural and Evolutionary Computation (NEC) - Assignment 2
####Wiktor Samulski

import numpy as np
import random
import os
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict


RANDOM_SEED = 42
#########JSSP instance and Chromosome Classes
class JSSPInstance:
#####Represents a Job Shop Scheduling Problem instance

    def __init__(self, jobs: List[List[Tuple[int, int]]], name: str = ""):
        self.jobs = jobs
        self.name = name
        self.num_jobs = len(jobs)
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
        genes = []
        for job_id, job in enumerate(instance.jobs):
            genes.extend([job_id] * len(job))
        random.shuffle(genes)
        return cls(genes, instance)

    @property
    def fitness(self) -> int:
        if self._fitness is None:
            self._fitness, self._schedule = self._calculate_makespan()
        return self._fitness

    @property
    def schedule(self) -> Dict:
        if self._schedule is None:
            self._fitness, self._schedule = self._calculate_makespan()
        return self._schedule

    def _calculate_makespan(self) -> Tuple[int, Dict]:
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
    def tournament_selection(population: List[Chromosome], tournament_size: int = 3) -> Chromosome:
    tournament = random.sample(population, min(tournament_size, len(population)))
    return min(tournament, key=lambda c: c.fitness)


def roulette_wheel_selection(population: List[Chromosome]) -> Chromosome:
    max_fitness = max(c.fitness for c in population)
    inverted_fitness = [max_fitness - c.fitness + 1 for c in population]
    total = sum(inverted_fitness)
    if total == 0:
        return random.choice(population)
    probs = [f / total for f in inverted_fitness]
    return random.choices(population, weights=probs, k=1)[0]


def rank_selection(population: List[Chromosome]) -> Chromosome:
    sorted_pop = sorted(population, key=lambda c: c.fitness)
    n = len(sorted_pop)
    ranks = list(range(n, 0, -1))
    total = sum(ranks)
    probs = [r / total for r in ranks]
    return random.choices(sorted_pop, weights=probs, k=1)[0]

###Crossover methods
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
def swap_mutation(chromosome: Chromosome, rate: float = 0.1) -> Chromosome:
    if random.random() > rate:
        return chromosome.copy()
    genes = chromosome.genes.copy()
    i, j = random.sample(range(len(genes)), 2)
    genes[i], genes[j] = genes[j], genes[i]
    return Chromosome(genes, chromosome.instance)


def insertion_mutation(chromosome: Chromosome, rate: float = 0.1) -> Chromosome:
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
