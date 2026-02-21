import random

def process_death_logic(cells, toxin_level):
    survival_rate = (100 - toxin_level) / 100.0
    return [c for c in cells if random.random() < survival_rate]