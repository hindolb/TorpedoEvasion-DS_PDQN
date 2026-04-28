import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

#Environmental Hyperparameters
EPSILON1 = 1.1560594530854869
EPSILON2 = 1.3988588210079729
SCALING_FACTOR = 17.574657210036534
DECOY_REWARD = 323.54720558089986


class AuvEvasionEnv(gym.Env):
    metadata = {"render_modes": ["human", "console"]}

    def __init__(self, n_torpedoes=2, noise_level=0.05, render_mode=None):
        super(AuvEvasionEnv, self).__init__()
        
        self.n = n_torpedoes
        self.noise_level = noise_level
        self.render_mode = render_mode
        self.dt = 1.0  
        self.max_steps = 300
        
        #AUV- settings
        self.MAX_DEPTH = 250.0 
        self.MAX_SPEED = 15.0 * 0.514444  
        self.MIN_SPEED = 5.0 * 0.514444
        self.MAX_ACCEL = 0.5   
        self.MAX_YAW_RATE = np.deg2rad(15.0) 
        
        #Decoy settings
        # NOTE: ammo_mobile is now a simple list of ints
        self.MAX_MOBILE_DECOYS = 12
        self.MAX_HOVERING_DECOYS = 24
        self.DECOY_LIFE_STEPS = 60    
        self.DECOY_ACOUSTIC_MULTIPLIER = 1.5 
        self.mobile_angles = [0, 60, 120, 180, 240, 300] 
        
        #Torpedo settings
        self.torpedo_speed = 30.0 * 0.514444 
        self.hit_distance = 25.0             
        self.seeker_fov = np.deg2rad(120.0)  
        self.seeker_range = 3000.0           
        self.N_gain = 2.5                    
        
        #P-DQN ACTION SPACE:
        #Discrete(6)- ID of each decoy x Continuous(2)- AUV heading and AUV speed 
        self.action_space = spaces.Tuple((
            spaces.MultiDiscrete([37, 37]),
            spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        ))
        
        #State space
        self.state_dim = 20 # 16 AUV features + 4 Torpedo features
        self.observation_space = spaces.Box(low=-10000, high=10000, shape=(self.n, self.state_dim), dtype=np.float32)
        
        #Reward calculation settings
        self.prev_min_distance = None
        self.current_min_distance = None
        self.cummulitive_distance = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.decoy_id_counter = 0
        
        #Resetting AUV state- position, speed and decoy state
        self.auv_state = np.zeros(12)
        self.auv_state[0] = np.random.uniform(0.0, 100.0)
        self.auv_state[1] = np.random.uniform(0.0, 100.0)
        self.auv_state[2] = np.random.uniform(50.0, 150.0)
        self.auv_state[3] = self.MIN_SPEED
        
        self.ammo_mobile = [2, 2, 2, 2, 2, 2] # 2 per each of the 6 angles
        self.ammo_hovering = self.MAX_HOVERING_DECOYS
        self.active_decoys = []  
        
        #Resetting torpedoes state-orientation, position and velocity
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
        
        #Resetting reward calculation metrics
        self.current_min_distance = 0
        self.cummulitive_distance = 0
                    
        #Collect and return environment state observation at the star of episode
        return self._get_observation(), {}

    def step(self, action):
        # obtain the discrete and continuous action components from the parameter action
        # discrete action: 2- for each launch decisions- each contaains id- either 1- no launch, 2-13- mobile decoy IDS, 14-37- hovering decoy
        # continuous action: 2 parameters- AUV speed, AUV direction
        discrete_action, cont_params = action
        
        target_vel_norm = (cont_params[0] + 1.0) / 2.0
        target_vel = target_vel_norm * (self.MAX_SPEED - self.MIN_SPEED) + self.MIN_SPEED
        target_heading = cont_params[1] * math.pi
        
        # --- 1. AUV Kinematics ---
        #Old data
        current_vel = self.auv_state[3]
        current_psi = self.auv_state[8]
        current_theta = self.auv_state[9] # Pitch (from observation index 9)
        
        #Calculate acceleration and change in heading based on action output
        accel_u = np.clip(0.5 * (target_vel - current_vel), -self.MAX_ACCEL, self.MAX_ACCEL)
        diff_psi = (target_heading - current_psi + math.pi) % (2 * math.pi) - math.pi
        r_cmd = np.clip(0.5 * diff_psi, -self.MAX_YAW_RATE, self.MAX_YAW_RATE)
        
        #Compute new heading and speed based on calculation from new output
        self.auv_state[3] = np.clip(self.auv_state[3] + accel_u * self.dt, self.MIN_SPEED, self.MAX_SPEED)
        self.auv_state[8] = (self.auv_state[8] + r_cmd * self.dt + math.pi) % (2 * math.pi) - math.pi
        #Compute new position of AUV 
        #TODO: Why the 3rd dimensionalposition is not calculated?
        x_dot = self.auv_state[3] * math.cos(current_theta) * math.cos(self.auv_state[8])
        y_dot = self.auv_state[3] * math.cos(current_theta) * math.sin(self.auv_state[8])
        z_dot = -self.auv_state[3] * math.sin(current_theta)
        self.auv_state[0] += x_dot * self.dt
        self.auv_state[1] += y_dot * self.dt
        self.auv_state[2] = np.clip(self.auv_state[2] + z_dot * self.dt, 0.0, self.MAX_DEPTH)
        
        auv_pos = self.auv_state[0:3]
        auv_vel = np.array([x_dot, y_dot, z_dot])

        # --- 2. Decoy Deployment ---
        decoy_error = False

        # Execute Launch List
        for d_action in discrete_action:
            if d_action == 0:
                continue # No action for this slot    
            elif 13 <= d_action <= 36: # Hovering Decoys
                if self.ammo_hovering > 0:
                    self.ammo_hovering -= 1
                    self.decoy_id_counter += 1
                    self.active_decoys.append({
                        'id': self.decoy_id_counter, 'type': 'hovering',
                        'pos': auv_pos.copy(), 'vel': np.zeros(3), 'life': self.DECOY_LIFE_STEPS
                    })
                else:
                    decoy_error = True
            elif 1 <= d_action <= 12: # Mobile Decoys
                angle_idx = d_action%6
                if self.ammo_mobile[angle_idx] > 0:  
                    self.ammo_mobile[angle_idx] -= 1 
                    self.decoy_id_counter += 1
                    
                    pre_designated_angle = self.mobile_angles[angle_idx]
                    launch_heading = self.auv_state[8] + np.deg2rad(pre_designated_angle)
                    
                    #TODO: decoy may have velocity independent of AUV velocity 
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
        
        self.cummulitive_distance = self.current_min_distance / (self.prev_min_distance + 1e-6)

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
        # Hyperparameters for weighting (epsilon1, epsilon2) and penalty scaling
        # Note: Set these to your optimal tuned values
        epsilon1 = EPSILON1  
        epsilon2 = EPSILON2  
        scaling_factor = SCALING_FACTOR  

        # 1. Distance (R_dist) and Survival (R_survive) Rewards
        # Computed following equations 30 and 31 from Chung et al.
        R_dist = self.cummulitive_distance/self.prev_min_distance
        R_survive = self.max_steps / ((self.prev_min_distance + 1e-6) / self.torpedo_speed)

        # 2. Decoy Bonus (R_decoy_bonus)
        # Applied when a torpedo is deceived and drops pursuit of the AUV
        R_decoy_bonus = DECOY_REWARD * torpedoes_destroyed

        # 3. Hit Penalty (Equation 9)
        # Scales the positive reward of the last time step (R_{t-1}) and negates it
        R_hit_penalty = 0.0
        if hit_auv:
            # Fetch the previous time step's reward (R_{t-1}). 
            # If it's the very first step, fallback to the current step's unpenalized reward.
            R_t_minus_1 = getattr(self, 'prev_reward', (epsilon1 * R_dist) + (epsilon2 * R_survive))
            R_hit_penalty = -1.0 * scaling_factor * R_t_minus_1

        # 4. Total Reward calculation (Equation 8)
        R_t = (epsilon1 * R_dist) + (epsilon2 * R_survive) + R_decoy_bonus + R_hit_penalty

        # Track the current unpenalized step reward to serve as R_{t-1} in the next time step
        self.prev_reward = (epsilon1 * R_dist) + (epsilon2 * R_survive) + R_decoy_bonus

        return R_t
