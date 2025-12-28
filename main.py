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
