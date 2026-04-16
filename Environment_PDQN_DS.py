import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

#Hyperparameters
# --- Weights ---
# ==========================================
# REPLACE HYPERPARAMETERS AT THE TOP
# ==========================================
DECOY_PENALTY = 0.1      # Reduced from 1.0: Don't heavily punish early experimentation
HIT_PENALTY = 1500.0     # Catastrophic failure.
DECOY_REWARD = 300.0     # Increased from 50.0: Strong reinforcement for neutralizing a threat
EVASION_REWARD = 1500.0  # Ultimate goal.

# --- Absolute Values ---
DECOY_PENALTY = 1     # Small slap on the wrist for empty firing (was 10.0)
HIT_PENALTY = 1500.0     # Catastrophic failure. Must be the largest negative number
DECOY_REWARD = 50.0     # Strong reinforcement for successfully destroying a torpedo
EVASION_REWARD = 1500.0  # Ultimate goal. Must balance out the hit penalty

class AuvEvasionEnv(gym.Env):
    metadata = {"render_modes": ["human", "console"]}

    def __init__(self, n_torpedoes=2, noise_level=0.05, render_mode=None):
        super(AuvEvasionEnv, self).__init__()
        
        self.n = n_torpedoes
        self.noise_level = noise_level
        self.render_mode = render_mode
        self.dt = 1.0  
        self.max_steps = 300
        
        self.MAX_DEPTH = 250.0 
        self.MAX_SPEED = 15.0 * 0.514444  
        self.MIN_SPEED = 5.0 * 0.514444
        self.MAX_ACCEL = 0.5   
        self.MAX_YAW_RATE = np.deg2rad(15.0) 
        
        # NOTE: ammo_mobile is now a simple list of ints
        self.MAX_MOBILE_DECOYS = 12
        self.MAX_HOVERING_DECOYS = 24
        self.DECOY_LIFE_STEPS = 60    
        self.DECOY_ACOUSTIC_MULTIPLIER = 1.5 
        self.mobile_angles = [0, 60, 120, 180, 240, 300] 
        
        self.torpedo_speed = 30.0 * 0.514444 
        self.hit_distance = 25.0             
        self.seeker_fov = np.deg2rad(120.0)  
        self.seeker_range = 3000.0           
        self.N_gain = 2.5                    
        
        # P-DQN ACTION SPACE: Discrete(6) x Continuous(4)
        self.action_space = spaces.Tuple((
            spaces.Discrete(6),
            spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        ))
        
        # New code
        self.state_dim = 20 # 16 AUV features + 4 Torpedo features
        # Shape is now (Number of Torpedoes, 20)
        self.observation_space = spaces.Box(low=-10000, high=10000, shape=(self.n, self.state_dim), dtype=np.float32)
        
        self.prev_min_distance = None
        self.current_min_distance = None
        self.cummulitive_distance = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.decoy_id_counter = 0
        
        self.auv_state = np.zeros(12)
        self.auv_state[0] = np.random.uniform(0.0, 100.0)
        self.auv_state[1] = np.random.uniform(0.0, 100.0)
        self.auv_state[2] = np.random.uniform(50.0, 150.0)
        self.auv_state[3] = self.MIN_SPEED
        
        self.ammo_mobile = [2, 2, 2, 2, 2, 2] # 2 per each of the 6 angles
        self.ammo_hovering = self.MAX_HOVERING_DECOYS
        self.active_decoys = []  
        
        self.torpedoes = []
        for i in range(self.n):
            dist = np.random.uniform(2000.0, 2500.0)
            heading = np.random.uniform(-math.pi, math.pi)
            elevation = np.random.uniform(-math.pi/6, math.pi/6)
            
            t_x = dist * math.cos(elevation) * math.cos(heading) + self.auv_state[0]
            t_y = dist * math.cos(elevation) * math.sin(heading) + self.auv_state[1]
            t_z = dist * math.sin(elevation) + self.auv_state[2]
            
            t_pos = np.array([t_x, t_y, t_z])
            direction = self.auv_state[0:3] - t_pos
            t_vel = (direction / np.linalg.norm(direction)) * self.torpedo_speed
            
            self.torpedoes.append({
                'id': i, 'pos': t_pos, 'vel': t_vel, 'active': True, 
                'prev_xi': None, 'target_decoy_id': None
            })

        active_torpedoes = [t for t in self.torpedoes if t['active']]
        self.prev_min_distance = min([np.linalg.norm(self.auv_state[0:3] - t['pos']) for t in active_torpedoes]) if active_torpedoes else 0.0 
        
        self.current_min_distance = 0
        self.cummulitive_distance = 0
                    
        return self._get_observation(), {}

    def step(self, action):
        # UNPACK P-DQN Hybrid Action
        discrete_action, cont_params = action
        
        target_vel_norm = (cont_params[0] + 1.0) / 2.0
        target_vel = target_vel_norm * (self.MAX_SPEED - self.MIN_SPEED) + self.MIN_SPEED
        target_heading = cont_params[1] * math.pi
        
        # --- 1. AUV Kinematics ---
        current_vel = self.auv_state[3]
        current_psi = self.auv_state[8]
        
        accel_u = np.clip(0.5 * (target_vel - current_vel), -self.MAX_ACCEL, self.MAX_ACCEL)
        diff_psi = (target_heading - current_psi + math.pi) % (2 * math.pi) - math.pi
        r_cmd = np.clip(0.5 * diff_psi, -self.MAX_YAW_RATE, self.MAX_YAW_RATE)
        
        self.auv_state[3] = np.clip(self.auv_state[3] + accel_u * self.dt, self.MIN_SPEED, self.MAX_SPEED)
        self.auv_state[8] = (self.auv_state[8] + r_cmd * self.dt + math.pi) % (2 * math.pi) - math.pi
        
        x_dot = self.auv_state[3] * math.cos(self.auv_state[8])
        y_dot = self.auv_state[3] * math.sin(self.auv_state[8])
        self.auv_state[0] += x_dot * self.dt
        self.auv_state[1] += y_dot * self.dt
        
        auv_pos = self.auv_state[0:3]
        auv_vel = np.array([x_dot, y_dot, 0.0])

        # --- 2. Decoy Deployment ---
        decoy_error = False
        decoy_configs = []
        
        # Build launch list based on discrete choice
        if discrete_action == 1:
            decoy_configs.append(('hovering', None))
        elif discrete_action == 2:
            idx1 = int(((cont_params[2] + 1.0) / 2.0) * 5.99)
            decoy_configs.append(('mobile', idx1))
        elif discrete_action == 3:
            decoy_configs.append(('hovering', None))
            decoy_configs.append(('hovering', None))
        elif discrete_action == 4:
            idx1 = int(((cont_params[2] + 1.0) / 2.0) * 5.99)
            idx2 = int(((cont_params[3] + 1.0) / 2.0) * 5.99)
            decoy_configs.append(('mobile', idx1))
            decoy_configs.append(('mobile', idx2))
        elif discrete_action == 5:
            decoy_configs.append(('hovering', None))
            idx1 = int(((cont_params[2] + 1.0) / 2.0) * 5.99)
            decoy_configs.append(('mobile', idx1))

        # Execute Launch List
        for d_type, angle_idx in decoy_configs:
            if d_type == 'hovering':
                if self.ammo_hovering > 0:
                    self.ammo_hovering -= 1
                    self.decoy_id_counter += 1
                    self.active_decoys.append({
                        'id': self.decoy_id_counter, 'type': 'hovering',
                        'pos': auv_pos.copy(), 'vel': np.zeros(3), 'life': self.DECOY_LIFE_STEPS
                    })
                else:
                    decoy_error = True
            
            elif d_type == 'mobile':
                if self.ammo_mobile[angle_idx] > 0:  # FIXED IndexError logic
                    self.ammo_mobile[angle_idx] -= 1 # FIXED IndexError logic
                    self.decoy_id_counter += 1
                    
                    pre_designated_angle = self.mobile_angles[angle_idx]
                    launch_heading = self.auv_state[8] + np.deg2rad(pre_designated_angle)
                    
                    decoy_v = np.array([self.auv_state[3] * math.cos(launch_heading), 
                                        self.auv_state[3] * math.sin(launch_heading), 0.0])
                    self.active_decoys.append({
                        'id': self.decoy_id_counter, 'type': 'mobile',
                        'pos': auv_pos.copy(), 'vel': decoy_v, 'life': self.DECOY_LIFE_STEPS
                    })
                else:
                    decoy_error = True                

        # Update active decoys
        alive_decoys = []
        for d in self.active_decoys:
            if d['type'] == 'mobile':
                d['pos'] += d['vel'] * self.dt
            d['life'] -= 1
            if d['life'] > 0:
                alive_decoys.append(d)
        self.active_decoys = alive_decoys

        # --- 3. Torpedo Seduction & Kinematics ---
        hit_auv = False
        torpedoes_destroyed = 0
        
        for torp in self.torpedoes:
            if not torp['active']: continue
            
            torp_heading_vec = torp['vel'] / (np.linalg.norm(torp['vel']) + 1e-6)
            visible_targets = []
            
            vec_to_auv = auv_pos - torp['pos']
            dist_to_auv = np.linalg.norm(vec_to_auv)
            if dist_to_auv < self.seeker_range:
                auv_dir = vec_to_auv / (dist_to_auv + 1e-6)
                angle_to_auv = np.arccos(np.clip(np.dot(torp_heading_vec, auv_dir), -1.0, 1.0))
                if angle_to_auv <= self.seeker_fov / 2.0:
                    intensity = 1.0 / (dist_to_auv**2 + 1e-6)
                    visible_targets.append({'type': 'auv', 'pos': auv_pos, 'vel': auv_vel, 'intensity': intensity})
                    
            for decoy in self.active_decoys:
                vec_to_decoy = decoy['pos'] - torp['pos']
                dist_to_decoy = np.linalg.norm(vec_to_decoy)
                if dist_to_decoy < self.seeker_range:
                    decoy_dir = vec_to_decoy / (dist_to_decoy + 1e-6)
                    angle_to_decoy = np.arccos(np.clip(np.dot(torp_heading_vec, decoy_dir), -1.0, 1.0))
                    if angle_to_decoy <= self.seeker_fov / 2.0:
                        intensity = self.DECOY_ACOUSTIC_MULTIPLIER / (dist_to_decoy**2 + 1e-6)
                        visible_targets.append({
                            'type': 'decoy', 'id': decoy['id'], 'pos': decoy['pos'], 
                            'vel': decoy['vel'], 'intensity': intensity
                        })

            target_pos = None
            target_vel = None
            
            if visible_targets:
                best_target = max(visible_targets, key=lambda t: t['intensity'])
                target_pos = best_target['pos']
                target_vel = best_target['vel']
                
                if best_target['type'] == 'decoy':
                    if np.linalg.norm(torp['pos'] - best_target['pos']) < self.hit_distance:
                        torp['active'] = False
                        target_id = best_target['id']
                        self.active_decoys = [d for d in self.active_decoys if d['id'] != target_id]
                        
                        if dist_to_auv < (self.hit_distance * 2.0):
                            hit_auv = True
                        else:
                            torpedoes_destroyed += 1
                        continue

            if dist_to_auv < self.hit_distance:
                hit_auv = True
                torp['active'] = False
                continue

            if torp['active'] and target_pos is not None:
                rel_pos = target_pos - torp['pos']
                rel_vel = target_vel - torp['vel']
                dist_to_target = np.linalg.norm(rel_pos)
                
                if dist_to_target > 0:
                    los_rate = np.cross(rel_pos, rel_vel) / (dist_to_target**2)
                    v_closing = -np.dot(rel_vel, rel_pos / dist_to_target)
                    
                    if v_closing > 0:
                        accel_dir = np.cross(los_rate, rel_pos / dist_to_target)
                        accel_cmd = self.N_gain * v_closing * accel_dir
                        
                        torp['vel'] += accel_cmd * self.dt
                        torp['vel'] = (torp['vel'] / np.linalg.norm(torp['vel'])) * self.torpedo_speed
                        
            torp['pos'] += torp['vel'] * self.dt

        self.current_step += 1
        all_destroyed = all(not t['active'] for t in self.torpedoes)
        termination = hit_auv or all_destroyed
        truncation = self.current_step >= self.max_steps

        active_torpedoes = [t for t in self.torpedoes if t['active']]
        if active_torpedoes:
            distances = [np.linalg.norm(self.auv_state[0:3] - t['pos']) for t in active_torpedoes]
            self.current_min_distance = min(distances)
        else:
            self.current_min_distance = 0 
        
        self.cummulitive_distance += self.current_min_distance / (self.prev_min_distance + 1e-6)

        reward = self.get_reward(decoy_error, hit_auv, all_destroyed, truncation, torpedoes_destroyed)
        #self.prev_min_distance = self.current_min_distance
            
        return self._get_observation(), reward, termination, truncation, {"hit": hit_auv, "won": all_destroyed}

    def _get_observation(self):
        x, y, z = self.auv_state[0:3]
        u, v, w = self.auv_state[3:6]
        phi, theta, psi = self.auv_state[6:9]
        p, q, r = self.auv_state[9:12]
        
        v_E = np.linalg.norm([u, v, w])
        
        obs_auv = [
            x / 5000.0, y / 5000.0, z / self.MAX_DEPTH, 
            v_E / self.MAX_SPEED, v / self.MAX_SPEED, u / self.MAX_SPEED,
            z / self.MAX_DEPTH, self.MAX_DEPTH / 1000.0, 
            phi / math.pi, theta / math.pi, psi / (2*math.pi),
            p, q, r / self.MAX_YAW_RATE, 
            sum(self.ammo_mobile) / float(self.MAX_MOBILE_DECOYS), 
            self.ammo_hovering / float(self.MAX_HOVERING_DECOYS)
        ]

        multi_obs = []

        for torp in self.torpedoes:
            if not torp['active']:
                torp_obs = [0.0, 0.0, 0.0, 1.0]
                multi_obs.append(obs_auv + torp_obs) 
                continue
                
            rel_pos = torp['pos'] - self.auv_state[0:3]
            true_dist = np.linalg.norm(rel_pos)
            l = true_dist * (1.0 + np.random.normal(0, self.noise_level))
            
            angle_to_torpedo = math.atan2(rel_pos[1], rel_pos[0])
            xi = (angle_to_torpedo - psi + math.pi) % (2 * math.pi) - math.pi 
            zeta = math.atan2(rel_pos[2], math.sqrt(rel_pos[0]**2 + rel_pos[1]**2 + 1e-6))
            
            omega = 0.0 if torp['prev_xi'] is None else ((xi - torp['prev_xi'] + math.pi) % (2 * math.pi) - math.pi) / self.dt
            torp['prev_xi'] = xi
            
            torp_obs = [xi / math.pi, zeta / (math.pi/2), omega / math.pi, l / 5000.0]
            
            # Concatenate AUV state with THIS specific torpedo's state
            multi_obs.append(obs_auv + torp_obs)
            
        return np.array(multi_obs, dtype=np.float32)


    def get_reward(self, decoy_error, hit_auv, all_destroyed, truncation, torpedoes_destroyed):
        # 1. Base step penalty acts as a ticking clock
        reward = -1.0 
    
        # 2. Continuous Distance Guidance ("Warmer/Colder" signal)
        active_torps = [t for t in self.torpedoes if t['active']]
        if active_torps:
            # Scale distance from 0 to 1 based on seeker range (3000m)
            safety_scores = [np.clip(np.linalg.norm(self.auv_state[0:3] - t['pos']) / 3000.0, 0.0, 1.0) for t in active_torps]
            # Multiplier of 5.0 gives a noticeable +0 to +5 reward per step for staying far away
            reward += 2.0 * (sum(safety_scores) / len(safety_scores))

        # 3. Small penalty for firing empty
        if decoy_error:
            reward -= DECOY_PENALTY
        
        # 4. Immediate massive reward for using a decoy successfully
        reward += (DECOY_REWARD * torpedoes_destroyed)
    
        # 5. Terminal Outcomes (Strictly separated)
        if hit_auv:
            reward -= HIT_PENALTY
        elif all_destroyed:
            reward += EVASION_REWARD
        elif truncation:
            reward += 500.0  # Small reward for surviving the full duration, but less than actual evasion
        
        return reward
'''
    def get_reward(self, decoy_error, hit_auv, all_destroyed, truncation, torpedoes_destroyed):
        # 1. Base step penalty acts as a ticking clock.
        # This replaces the positive time_reward and forces fast action.
        step_penalty = -1.0 
        
        # 2. Intermediate success (Torpedo hits a decoy)
        decoy_reward = 50.0 * torpedoes_destroyed
        
        # 3. Terminal Outcomes (STRICTLY SEPARATED)
        terminal_reward = 0.0
        if hit_auv:
            terminal_reward = -1500.0  # Match your absolute value
        elif all_destroyed:
            terminal_reward = 1500.0   # Match your absolute value
        elif truncation:
            terminal_reward = 0.0      # NO REWARD for timing out. They must fight!

        # Total pure reward
        reward = step_penalty + decoy_reward + terminal_reward
        
        return reward
'''
'''
    def get_reward(self, decoy_error, hit_auv, all_destroyed, truncation, torpedoes_destroyed):        
        # FIXED: ZeroDivisionError safety 1e-6 added
        time_reward = (self.max_steps)/((self.prev_min_distance + 1e-6)/self.torpedo_speed) 
        
        hit_penalty = HIT_PENALTY if hit_auv else 0
        #decoy_reward = EPSILON5 * DECOY_REWARD * torpedoes_destroyed
        evasion_reward = EVASION_REWARD if (not hit_auv and (all_destroyed or truncation)) else 0
        
        return time_reward - hit_penalty + evasion_reward
'''
'''
    def get_reward(self, decoy_error, hit_auv, all_destroyed, truncation, torpedoes_destroyed):
        distance_reward = 0.0
        active_torps = [t for t in self.torpedoes if t['active']]
        
        if active_torps:
            # Calculate a bounded safety score (0 to 1) for EACH torpedo independently
            safety_scores = [np.clip(np.linalg.norm(self.auv_state[0:3] - t['pos']) / 3000.0, 0.0, 1.0) for t in active_torps]
            
            # Average the safety scores. 
            # If 1 torp is 100m away (0.03) and 1 torp is 2000m away (0.66), average is ~0.34
            distance_reward = sum(safety_scores) / len(safety_scores)
        decoy_penalty = EPSILON2 * DECOY_PENALTY if decoy_error else 0   
        
        # FIXED: ZeroDivisionError safety 1e-6 added
        time_reward = EPSILON3 * (self.max_steps)/((self.prev_min_distance + 1e-6)/self.torpedo_speed) 
        
        hit_penalty = EPSILON4 * HIT_PENALTY if hit_auv else 0
        decoy_reward = EPSILON5 * DECOY_REWARD * torpedoes_destroyed
        evasion_reward = EPSILON6 * EVASION_REWARD if (not hit_auv and (all_destroyed or truncation)) else 0
        
        return distance_reward + time_reward - decoy_penalty - hit_penalty + decoy_reward + evasion_reward
'''
'''
    def get_reward(self, decoy_error, hit_auv, all_destroyed, truncation, torpedoes_destroyed):
        # 1. Base step penalty (Replaces the broken time_reward equation)
        # This acts as a ticking clock, forcing the agent to end the episode quickly.
        step_penalty = -1.0 
        
        # 2. Safe, Bounded Distance Breadcrumbs (You wrote this correctly!)
        safe_distance_reward = 0.0
        #active_torps = [t for t in self.torpedoes if t['active']]
        
        #if active_torps:
        #    safety_scores = [np.clip(np.linalg.norm(self.auv_state[0:3] - t['pos']) / 3000.0, 0.0, 1.0) for t in active_torps]
        #    safe_distance_reward = sum(safety_scores) / len(safety_scores)
        
        # 3. Intermediate success (Torpedo hits a decoy)
        decoy_reward = 50.0 * torpedoes_destroyed
        
        # 4. Terminal Outcomes (STRICTLY SEPARATED)
        terminal_reward = 0.0
        if hit_auv:
            terminal_reward = -500.0   # Catastrophic failure
        elif all_destroyed:
            terminal_reward = 500.0    # Ultimate Success (Threats neutralized)
        elif truncation:
            terminal_reward = 100.0    # Moderate Success (Survived 300 steps, but threats remain)

        # Total reward (No EPSILON scaling needed, values are perfectly balanced)
        reward = step_penalty + safe_distance_reward + decoy_reward + terminal_reward
        
        return reward
'''

'''
    def get_reward(self,decoy_error, hit_auv, all_destroyed, truncation, torpedoes_destroyed):
        distance_reward = EPSILON1 * (self.cummulitive_distance)      
        decoy_penalty = EPSILON2 * DECOY_PENALTY if decoy_error else 0   
        
        # FIXED: ZeroDivisionError safety 1e-6 added
        time_reward = EPSILON3 * (self.max_steps)/((self.prev_min_distance + 1e-6)/self.torpedo_speed) 
        
        hit_penalty = EPSILON4 * HIT_PENALTY if hit_auv else 0
        decoy_reward = EPSILON5 * DECOY_REWARD * torpedoes_destroyed
        evasion_reward = EPSILON6 * EVASION_REWARD if (not hit_auv and (all_destroyed or truncation)) else 0
        
        return distance_reward + time_reward - decoy_penalty - hit_penalty + decoy_reward + evasion_reward


    def get_reward(self, decoy_error, hit_auv, all_destroyed, truncation, torpedoes_destroyed):
        # 1. Base time penalty: Encourages AUV to neutralize the threat quickly
        step_penalty = -1.0 
        
        # 2. Decoy empty fire penalty
        decoy_penalty = 2.0 if decoy_error else 0.0
        
        # 3. Intermediate success (Torpedo hits a decoy)
        decoy_reward = 50.0 * torpedoes_destroyed
        
        # 4. Terminal Outcomes
        terminal_reward = 0.0
        if hit_auv:
            terminal_reward = -500.0   # Catastrophic failure
        elif all_destroyed:
            terminal_reward = 500.0    # Ultimate Success (Threat neutralized)
        elif truncation:
            terminal_reward = 100.0    # Moderate Success (Survived, but threat still exists)

        # Total reward for this step
        reward = step_penalty - decoy_penalty + decoy_reward + terminal_reward
        
        return reward
'''