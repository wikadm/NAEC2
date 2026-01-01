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
