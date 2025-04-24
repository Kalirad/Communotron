"""
Module: communotron
Description: This module defines the Community class and its methods for simulating population dynamics.
Author: Ata Kalirad
Version: 1.0
Date: 20.01.2025
"""

import numpy as np
import random as rnd
import pandas as pd
from copy import deepcopy
import shortuuid


class Community(object):

    def __init__(self, grid_size, source_center, source_diameter, decline_rate, resource_cycle=0, pred_par_a = 0.01, pred_par_b = 0.1, delta=75, time_lim=300, resource_lim=0.4):
        """
        Initializes the class with the given parameters.
            grid_size (int): The size of the grid.
            source_center (tuple): The coordinates of the source center.
            source_diameter (float): The diameter of the source.
            decline_rate (float): The rate at which the resource declines.
            resource_cycle (int, optional): The cycle of the resource. Defaults to 0.
            pred_par_a (float, optional): Parameter 'a' for the predator model. Defaults to 0.01.
            pred_par_b (float, optional): Parameter 'b' for the predator model. Defaults to 0.1.
            delta (int, optional): The delta value for the model. Defaults to 75.
            time_lim (int, optional): The time limit for the simulation. Defaults to 300.
            resource_lim (float, optional): The resource limit. Defaults to 0.4.
        """
        self.grid_size = grid_size
        self.source_center = source_center
        self.source_diameter = source_diameter
        self.decline_rate = decline_rate
        self.resource_cycle = resource_cycle
        self.pred_par_a = pred_par_a
        self.pred_par_b = pred_par_b
        self.delta = delta
        self.resource_lim = resource_lim
        self.time = 0
        self.time_lim = time_lim
        self.migrants = []
        self.migrants_cum = {'A': 0, 'B': 0, 'C': 0}
        self.moore_neighborhood = np.array([[-1, -1], [-1, 0], [-1, 1],
                                   [0, -1],           [0, 1],
                                   [1, -1], [1, 0], [1, 1]])
    
        self.directions = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
        self.ind_fec = {}
        self.trace = {}
        
    @staticmethod
    def distribute_eggs_uniformly(m, L=24):
        """
        Distributes the eggs uniformly.
            m (list): The list of eggs.
            L (int, optional): The number of hours in a day. Defaults to
            24.
        Returns:
            np.array: The distributed eggs.
        """
        assert type(m) == list
        dist = []
        for i in m:
            q, r = divmod(i, L)
            daily_dist = [q + 1] * r + [q] * (L - r)
            dist += daily_dist
        rnd.shuffle(dist)
        return np.array(dist)

    def add_features(self, mf_probs, dev_pars, surv_pars, fec_data, age_lim):
        """
        Adds the features to the class.
            mf_probs (dict): The probabilities
            dev_pars (dict): The parameters
            fec_data (dict): The fecundity data
            age_lim (dict): The age limit
        """
        self.mf_probs = mf_probs
        self.dev_pars = dev_pars
        self.surv_pars = surv_pars
        self.fec_data = fec_data
        self.age_lim = age_lim

    @staticmethod
    def generate_uniform_coordinates_around_center(n, num_points, radius):
        """
        Generates uniform coordinates around the center.
            n (int): The size of the grid.
            num_points (int): The number of points.
            radius (int): The radius.
        Returns:
            np.array: The generated coordinates. 
        """
        center_x, center_y = (n - 1) / 2, (n - 1) / 2
        coordinates = []
        for _ in range(num_points):
            r = radius * np.sqrt(np.random.uniform(0, 1))
            theta = np.random.uniform(0, 2 * np.pi)
            x = center_x + r * np.cos(theta)
            y = center_y + r * np.sin(theta)
            x = int(np.clip(np.round(x), 0, n - 1))
            y = int(np.clip(np.round(y), 0, n - 1))
            coordinates.append((x, y))
        return np.array(coordinates)

    def add_strains(self, strains, dev_state, mf_state, age, sex, positions=False, centralized=True):
        """
        Adds the strains to the class.
            strains (list): The list of strains.
            dev_state (list): The list of developmental states.
            mf_state (list): The list of mf states.
            age (list): The list of ages.
            sex (list): The list of sexes.
            positions (np.array, optional): The positions. Defaults to
            False.
            centralized (bool, optional): Whether the positions are
            centralized. Defaults to True. 
        """
        assert len(strains) == len(dev_state) == len(mf_state) == len(age)
        if isinstance(positions, np.ndarray):
            self.positions = positions
        else:
            if centralized:
                self.positions = self.generate_uniform_coordinates_around_center(self.grid_size, len(strains), 10)
            else:
                self.positions = np.random.randint(0, self.grid_size, size=(len(strains), 2))
        self.strains = strains
        self.dev_state = dev_state
        self.mf_state = mf_state
        self.age = age
        self.uid = np.array([shortuuid.uuid() for i in range(len(strains))])
        self.sex = sex
        parents_idx = np.where((dev_state == 2) & (sex != 'M'))[0]
        if len(parents_idx) > 0:
            for id in parents_idx:
                self.ind_fec[self.uid[id]] = self.distribute_eggs_uniformly(self.fec_data[strains[id]])


    def increase_time(self):
        """ 
        Increases the time by 1.
        """
        self.time += 1

    def increase_age(self):
        """
        Increases the age vector by 1.
        """
        self.age += 1

    def set_max_steps(self, max_steps):
        """
        Sets the maximum steps.
            max_steps (int): The maximum steps. 
        """
        assert type(max_steps) is int
        self.max_steps = max_steps

    def cal_dev_prob(self):
        """
        Calculates the developmental probabilities.
        Returns:
            np.array: The probabilities. 
        """
        probabilities = []
        k = 0.1
        for i,j,l in zip(self.strains, self.dev_state, self.age):
            if j==2 or j == 3:
                probabilities.append(0.)
            else:
                midpoint = self.dev_pars[i][j]
                probabilities.append(1 / (1 + np.exp(-k * (l - midpoint))))
        return np.array(probabilities)
    

    def cal_death_prob(self):
        probabilities = []
        k = 0.1
        for i,j,l in zip(self.strains, self.dev_state, self.age):
            if j == 2 or j == 4 or j == 5:
                midpoint = self.surv_pars[j]
                probabilities.append(1 / (1 + np.exp(-k * (l - midpoint))))
            else:
               probabilities.append(0.)
        return np.array(probabilities)

    def update_mf_state(self):
        """
        Updates the mf state.
        """
        for i in range(len(self.dev_state)):
            if self.old_dev_state[i] in [1, 4] and self.dev_state[i] == 2 and self.mf_state[i] == 0:
                strain = self.strains[i]
                if np.random.rand() < self.mf_probs.get(strain, 0):
                    self.mf_state[i] = 1  

    def update_survival(self):
        """
        Updates the survival state.
        """
        probabilities = self.cal_death_prob()
        rands = np.random.rand(len(probabilities))
        for i in range(len(self.dev_state)):
            if rands[i] < probabilities[i]:
                self.dev_state[i] = 3
                self.age[i] = 0

    def update_dev_state(self):
        """
        Updates the developmental state. 
        """
        probabilities = self.cal_dev_prob()
        current_gradient = self.current_gradient
        self.old_dev_state = deepcopy(self.dev_state)
        positions = self.positions[np.where((self.dev_state != 0) & (self.dev_state != 3))]
        rands = np.random.rand(len(probabilities))
        for i in range(len(self.dev_state)):
            if rands[i] < probabilities[i]:
                if self.dev_state[i] == 0:
                    pos = self.positions[i]
                    if current_gradient[pos[0], pos[1]] < self.resource_lim:
                        self.dev_state[i] = 5
                    else:
                        self.dev_state[i] += 1
                    self.age[i] = 0
                elif self.dev_state[i] == 1:
                    neighbor_positions_raw = [n+self.positions[i] for n in self.moore_neighborhood]
                    neighbor_positions = np.array([n for n in neighbor_positions_raw if (0<=n[0]<self.grid_size  and 0<=n[1]<self.grid_size)])
                    neighborhood_full = np.array([np.any(np.all(row == positions, axis=1)) for row in neighbor_positions]).all()
                    if neighborhood_full:
                        self.dev_state[i] = 4
                    else:
                        self.dev_state[i] = 2
                        self.ind_fec[self.uid[i]] = self.distribute_eggs_uniformly(self.fec_data[self.strains[i]])
                    self.age[i] = 0
                elif self.dev_state[i] == 4:
                    pos = self.positions[i]
                    if (current_gradient[pos[0], pos[1]] >= self.resource_lim):
                        neighbor_positions_raw = [n+self.positions[i] for n in self.moore_neighborhood]
                        neighbor_positions = np.array([n for n in neighbor_positions_raw if (0<=n[0]<self.grid_size  and 0<=n[1]<self.grid_size)])
                        neighborhood_full = np.array([np.any(np.all(row == positions, axis=1)) for row in neighbor_positions]).all()
                        if not neighborhood_full:
                            self.dev_state[i] = 2
                            self.ind_fec[self.uid[i]] = self.distribute_eggs_uniformly(self.fec_data[self.strains[i]])
                            self.age[i] = 0
                elif self.dev_state[i] == 5:
                    pos = self.positions[i]
                    if (current_gradient[pos[0], pos[1]] >= self.resource_lim):
                        neighbor_positions_raw = [n+self.positions[i] for n in self.moore_neighborhood]
                        neighbor_positions = np.array([n for n in neighbor_positions_raw if (0<=n[0]<self.grid_size  and 0<=n[1]<self.grid_size)])
                        neighborhood_full = np.array([np.any(np.all(row == positions, axis=1)) for row in neighbor_positions]).all()
                        if not neighborhood_full:
                            self.dev_state[i] = 1
                            self.ind_fec[self.uid[i]] = self.distribute_eggs_uniformly(self.fec_data[self.strains[i]])
                            self.age[i] = 0
                
    def generate_gradient(self):
        """
        Generates the gradient.
        Returns:
            np.array: The generated gradient
        """
        rows = cols = self.grid_size
        gradient = np.zeros((rows, cols))
        radius = self.source_diameter / 2
        for i in range(rows):
            for j in range(cols):
                distance = np.sqrt((i - self.source_center[0])**2 + (j - self.source_center[1])**2)
                if distance <= radius:
                    gradient[i, j] = 1  # Max concentration inside the source
                else:
                    gradient[i, j] = np.exp(-self.decline_rate * (distance - radius))  # Decline outside the source
        return self.resource_level * gradient

    def initialize_history(self):
        """
        Initializes the history dataframe.
        """
        history = pd.DataFrame()
        total_dauer = len(np.where(self.dev_state==4)[0])
        total_E = len(np.where(self.dev_state==0)[0])
        total_J = len(np.where(self.dev_state==1)[0])
        total_A = len(np.where(self.dev_state==2)[0])
        total_dead = len(np.where(self.dev_state==3)[0])
        total_dauer = len(np.where(self.dev_state==4)[0])
        total_AJ = len(np.where(self.dev_state==5)[0])
        total_cum_mig = sum(self.migrants_cum.values())
        for i in ['A', 'B', 'C']:
            strain_list = self.dev_state[np.where(self.strains==i)]
            E_count = len(np.where(strain_list==0)[0])
            J_count = len(np.where(strain_list==1)[0])
            A_count = len(np.where(strain_list==2)[0])
            Dead_count = len(np.where(strain_list==3)[0])
            Dauer_count = len(np.where(strain_list==4)[0])
            AJ_count = len(np.where(strain_list==5)[0])
            fE = E_count/total_E if total_E > 0 else 0
            fJ = J_count/total_J if total_J > 0 else 0
            fA = A_count/total_A if total_A > 0 else 0
            fDead = Dead_count/total_dead if total_dead > 0 else 0
            fDauer = Dauer_count/total_dauer if total_dauer > 0 else 0
            fAJ = AJ_count/total_AJ if total_AJ > 0 else 0
            fcum_mig = self.migrants_cum[i]/total_cum_mig if total_cum_mig > 0 else 0
            temp = pd.DataFrame.from_dict({'E': [E_count],
                                            'J': [J_count],
                                            'A': [A_count],
                                            'Dead': [Dead_count],
                                            'Dauer': [Dauer_count],
                                            'AJ': [AJ_count],
                                            'fE': [fE],
                                            'fJ': [fJ],
                                            'fA': [fA],
                                            'fDead': [fDead],
                                            'fDauer': [fDauer],
                                            'fAJ': [fAJ],
                                            'killed': [0],
                                            'cum_migrated': [self.migrants_cum[i]],
                                            'cum_migrated_f': [fcum_mig]})
            temp['Strain'] = i
            temp['Time'] = self.time
            temp['Resource'] = self.resource_level
            history = pd.concat([history, temp])
        self.history = deepcopy(history)
    
    def update_history(self):
        """
        Updates the history dataframe.
        """
        killed = [str(i[0]) for i in self.killed_worms.values()]
        total_E = len(np.where(self.dev_state==0)[0])
        total_J = len(np.where(self.dev_state==1)[0])
        total_A = len(np.where(self.dev_state==2)[0])
        total_dead = len(np.where(self.dev_state==3)[0])
        total_dauer = len(np.where(self.dev_state==4)[0])
        total_AJ = len(np.where(self.dev_state==5)[0])
        total_cum_mig = sum(self.migrants_cum.values())
        for i in ['A', 'B', 'C']:
            strain_killed = killed.count(i)
            strain_list = self.dev_state[np.where(self.strains==i)]
            E_count = len(np.where(strain_list==0)[0])
            J_count = len(np.where(strain_list==1)[0])
            A_count = len(np.where(strain_list==2)[0])
            Dead_count = len(np.where(strain_list==3)[0])
            Dauer_count = len(np.where(strain_list==4)[0])
            AJ_count = len(np.where(strain_list==5)[0])
            fE = E_count/total_E if total_E > 0 else 0
            fJ = J_count/total_J if total_J > 0 else 0
            fA = A_count/total_A if total_A > 0 else 0
            fDead = Dead_count/total_dead if total_dead > 0 else 0
            fDauer = Dauer_count/total_dauer if total_dauer > 0 else 0
            fAJ = AJ_count/total_AJ if total_AJ > 0 else 0
            fcum_mig = self.migrants_cum[i]/total_cum_mig if total_cum_mig > 0 else 0
            temp = pd.DataFrame.from_dict({'E': [E_count],
                                        'J': [J_count],
                                        'A': [A_count],
                                        'Dead': [Dead_count],
                                        'Dauer': [Dauer_count],
                                        'AJ': [AJ_count],
                                        'fE': [fE],
                                        'fJ': [fJ],
                                        'fA': [fA],
                                        'fDead': [fDead],
                                        'fDauer': [fDauer],
                                        'fAJ': [fAJ],
                                        'killed': [strain_killed],
                                        'cum_migrated': [self.migrants_cum[i]],
                                        'cum_migrated_f': [fcum_mig]})
            temp['Strain'] = i
            temp['Time'] = self.time
            temp['Resource'] = self.resource_level
            self.history = pd.concat([self.history, temp])

    @property
    def resource_level(self):
        """
        Returns the resource level. 
        """
        if self.resource_cycle == 0:
            return 1.0
        else:
            if (self.time_lim > 0) and (self.time > self.time_lim):
                return 0.0
            else:
                omega = 2*np.pi/self.resource_cycle
                return (1 + np.sin(omega*(self.time - self.delta)))/2
            

    @property
    def current_gradient(self):
        """
        Returns the current gradient 
        """
        return self.resource_level*self.generate_gradient()
    
    def remove_old_fec(self):
        """
        Removes the old fecundity.
        """
        dead_keys = [i for i,j,k in zip(self.uid, self.age, self.strains) if j > self.age_lim[k]]
        for key in dead_keys:
            if key in self.ind_fec:
                del self.ind_fec[key]

    def reproduce(self):
        """
        Reproduces the worms. 
        """
        moore_offsets_eggs = np.array([
                [-1, -1], [-1, 0], [-1, 1],
                [0, -1],          [0, 1],
                [1, -1], [1, 0], [1, 1]])
        parents_idx_raw = np.where((self.dev_state==2) & (self.sex != 'M'))[0]
        parents_idx = [i for i in parents_idx_raw if self.age[i] < self.age_lim[self.strains[i]]]
        if len(parents_idx) > 0:
            egg_strains = []
            egg_positions = []
            egg_sex = []
            for idx in parents_idx:
                daily_m  = self.ind_fec[self.uid[idx]][int(self.age[idx])]
                if daily_m > 0:
                    np.random.shuffle(moore_offsets_eggs)
                    copies_made = 0
                    for offset in moore_offsets_eggs:
                        if copies_made >= daily_m:
                            break
                        else:
                            neighbor_pos = self.positions[idx] + offset
                            if (0 <= neighbor_pos[0] < self.grid_size and 0 <= neighbor_pos[1] < self.grid_size):
                                egg_strains.append(self.strains[idx])
                                egg_positions.append(neighbor_pos)
                                if self.strains[idx] == 'C':
                                    if np.random.random() < 0.5:
                                        egg_sex.append('M')
                                    else:
                                        egg_sex.append('F')
                                else:
                                    egg_sex.append('H')
                                copies_made += 1
            if len(egg_strains) > 0:
                egg_dev_state = np.zeros(len(egg_strains))
                egg_mf_state = np.zeros(len(egg_strains))
                egg_age =  np.zeros(len(egg_strains))
                egg_uid = np.array([shortuuid.uuid() for i in range(len(egg_strains))])
                self.strains = deepcopy(np.append(self.strains, egg_strains))
                self.dev_state = deepcopy(np.append(self.dev_state, egg_dev_state))
                self.mf_state = deepcopy(np.append(self.mf_state, egg_mf_state))
                self.age = deepcopy(np.append(self.age, egg_age))
                self.sex = deepcopy(np.append(self.sex, egg_sex))
                self.positions = deepcopy(np.vstack([self.positions, np.array(egg_positions)]))
                self.uid = deepcopy(np.append(self.uid, egg_uid))

    @property
    def pop_density(self):
        """
        Returns the population density. 
        """
        positions =  self.positions[np.where(self.dev_state != 3)]
        x_coords, y_coords = positions[:, 0], positions[:, 1]
        frequency_grid, _, _ = np.histogram2d(x_coords, y_coords, bins=self.grid_size, range=[[0, self.grid_size], [0, self.grid_size]])
        if frequency_grid.sum() !=0:
            return frequency_grid / frequency_grid.sum()
        else:
            return frequency_grid
    
    def random_walk(self):
        """
        Performs the random walk.
        """
        self.killed_worms = {}
        gradient = self.current_gradient
        pop_density = self.pop_density
        # Exclude eggs (0), dead adults (3), and arrested juveniles (5) from movement
        movers_idx = np.where((self.dev_state != 0) & (self.dev_state != 3) & (self.dev_state != 5))[0]
        num_entities = len(movers_idx)
        movements = {i:[self.positions[i]] for i in movers_idx}
        if num_entities > 0:
            new_positions = deepcopy(self.positions)
            to_remove = set()
            dauer_exit = {}
            for i in range(1, self.max_steps):
                active_entities = [idx for idx in movers_idx if (idx not in to_remove)]
                if len(active_entities) == 0:
                    break
                else:
                    for idx in active_entities:
                        if self.dev_state[idx] == 4:
                            possible_moves = [new_positions[idx] + direction for direction in self.directions]
                            move_gradients = [pop_density[move[0], move[1]] if 0 <= move[0] < self.grid_size and 0 <= move[1] < self.grid_size else 0 for move in possible_moves]
                            probabilities = 1 / (np.array(move_gradients) + 1e-5) 
                        else:
                            possible_moves = [new_positions[idx] + direction for direction in self.directions]
                            possible_moves = [move for move in possible_moves if 0 <= move[0] < self.grid_size and 0 <= move[1] < self.grid_size]
                            move_gradients = [gradient[move[0], move[1]] if 0 <= move[0] < self.grid_size and 0 <= move[1] < self.grid_size else 0 for move in possible_moves]
                            probabilities = np.array(move_gradients) + 1e-5
                        # Normalize probabilities
                        probabilities /= probabilities.sum()
                        # Choose a move
                        chosen_move_idx = np.random.choice(len(possible_moves), p=probabilities)
                        chosen_move = possible_moves[chosen_move_idx]
                        if (chosen_move[0] < 0 or chosen_move[0] >= self.grid_size or chosen_move[1] < 0 or chosen_move[1] >= self.grid_size):
                            assert self.dev_state[idx] == 4
                            to_remove.add(idx)
                            dauer_exit[idx] = (self.strains[idx], self.dev_state[idx], self.age[idx], self.mf_state[idx], self.uid[idx])
                        # predation during movement
                        if self.dev_state[idx] == 2 and self.mf_state[idx] == 1:
                            other_entities_in_pos = np.where((new_positions == chosen_move).all(axis=1))[0]
                            # killing of juveniles, dauers, and arrested juveniles
                            potential_preys = [prey for prey in other_entities_in_pos if ((self.dev_state[prey] in {1, 4, 5}) and self.strains[prey] != self.strains[idx])]
                            if potential_preys:
                                pred_par = self.pred_par_b * np.exp(-self.pred_par_a * self.age[idx])
                                rands = np.random.rand(len(potential_preys))
                                for prey, rand in zip(potential_preys, rands):
                                    if rand < pred_par:
                                        to_remove.add(prey)
                                        self.killed_worms[prey] = (self.strains[prey], self.dev_state[prey], self.age[prey], self.mf_state[prey])
                        new_positions[idx] = chosen_move
                        movements[idx].append(chosen_move)
            temp = [i[0] for i in list(dauer_exit.values())]
            for i in ['A', 'B', 'C']:
                self.migrants_cum[i] += temp.count(i)
            self.migrants.append(dauer_exit) 
            if to_remove:
                self.positions = np.delete(self.positions, list(to_remove), axis=0)
                self.strains = np.delete(self.strains, list(to_remove))
                self.dev_state = np.delete(self.dev_state, list(to_remove))
                self.mf_state = np.delete(self.mf_state, list(to_remove))
                self.age = np.delete(self.age, list(to_remove))
                self.sex = np.delete(self.sex, list(to_remove))
                self.uid = np.delete(self.uid, list(to_remove))
            else:
                self.positions =  deepcopy(new_positions)
            self.trace[self.time] = movements

    def save_state(self, directory):
        """
        Saves the state.
            directory (str): The directory.  
        """
        self.history.to_csv(directory + 'output' + shortuuid.uuid() + '.csv')

    def simulate(self, time):
        """
        Simulates the model.
            time (int): The time. 
        """
        self.initialize_history()
        for _ in range(time):
            self.random_walk()
            self.reproduce()
            self.update_dev_state()
            self.update_mf_state()
            self.update_survival()
            self.remove_old_fec()
            self.update_history()
            self.increase_age()
            self.increase_time()



            