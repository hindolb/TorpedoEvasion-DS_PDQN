import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math


# ── Reward weights ────────────────────────────────────────────────────────────
# New reward formulation:
#   R_t = ε₁·R_dist + ε₂·R_survive + R_decoy_bonus + R_hit_penalty
EPSILON1       = 0.8    # weight for distance shaping term  (ε₁)
EPSILON2       = 1.0    # weight for per-step survival term (ε₂)
SCALING_FACTOR = 1.0    # scales prev_reward in the hit penalty
DECOY_REWARD   = 50.0   # bonus per torpedo destroyed by a decoy this step


rng = np.random.default_rng(seed=58)


class AuvEvasionEnv(gym.Env):
    """
    AUV torpedo evasion environment.

    State space  : (n_torpedoes, 20)  — per-torpedo combined AUV+torpedo state
    Action space : Tuple( MultiDiscrete([37, 37]),  Box(low=-1, high=1, shape=(2,)) )
                       discrete: two decoy IDs in {0, 1..12 mobile, 13..36 hovering}
                       continuous: (speed_norm, heading_norm), both in [-1, 1]
    """

    metadata = {"render_modes": ["human", "console"]}

    def __init__(self, n_torpedoes: int = 2,
                 noise_level: float = 0.05,
                 render_mode=None,
                 min_spawn_dist: float = 1500.0,
                 max_spawn_dist: float = 2400.0,
                 mode: str = "training"):
        super().__init__()

        self.n             = n_torpedoes
        self.noise_level   = noise_level
        self.render_mode   = render_mode
        self.dt            = 1.0
        self.max_steps     = 300

        # AUV settings
        self.MAX_DEPTH     = 250.0
        self.MAX_SPEED     = 15.0 * 0.514444   # knots -> m/s
        self.MIN_SPEED     = 5.0  * 0.514444
        self.MAX_ACCEL     = 0.5
        self.MAX_YAW_RATE  = np.deg2rad(15.0)

        # Decoy settings
        self.MAX_MOBILE_DECOYS         = 12
        self.MAX_HOVERING_DECOYS       = 24
        self.DECOY_LIFE_STEPS          = 60
        self.DECOY_ACOUSTIC_MULTIPLIER = 1.5
        self.mobile_angles             = [0, 60, 120, 180, 240, 300]

        # Torpedo settings
        self.torpedo_speed = 30.0 * 0.514444
        self.hit_distance  = 2.5
        self.seeker_fov    = np.deg2rad(120.0)
        self.seeker_range  = 1000.0
        self.N_gain        = 2.5

        # Spawn distance range (used by reset)
        self.min_spawn_dist = min_spawn_dist
        self.max_spawn_dist = max_spawn_dist

        # Action space
        self.action_space = spaces.Tuple((
            spaces.MultiDiscrete([37, 37]),
            spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        ))

        # Observation space
        self.state_dim = 20  # 16 AUV features + 4 torpedo features
        self.observation_space = spaces.Box(
            low=-10000, high=10000,
            shape=(self.n, self.state_dim),
            dtype=np.float32,
        )

        # Reward calculation state
        self.prev_min_distance    = None
        self.current_min_distance = None
        self.initial_min_distance = None   
        self.prev_reward          = 0.0

    # ------------------------------------------------------------------
    # Episode reset
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step      = 0
        self.decoy_id_counter  = 0

        # AUV initial state (12 elements: x,y,z, u,v,w, phi,theta,psi, p,q,r)
        self.auv_state    = np.zeros(12)
        self.auv_state[0] = rng.uniform(0.0, 100.0)
        self.auv_state[1] = rng.uniform(0.0, 100.0)
        self.auv_state[2] = rng.uniform(50.0, 150.0)
        self.auv_state[3] = self.MIN_SPEED   # u (forward speed)

        # Decoy ammo
        self.ammo_mobile   = [2, 2, 2, 2, 2, 2]   # 2 per each of 6 launch angles
        self.ammo_hovering = self.MAX_HOVERING_DECOYS
        self.active_decoys = []

        # Spawn torpedoes
        self.torpedoes = []
        for i in range(self.n):
            dist      = rng.uniform(self.min_spawn_dist, self.max_spawn_dist)
            heading   = rng.uniform(-math.pi, math.pi)
            elevation = rng.uniform(-math.pi/6, math.pi/6)

            t_x = dist * math.cos(elevation) * math.cos(heading) + self.auv_state[0]
            t_y = dist * math.cos(elevation) * math.sin(heading) + self.auv_state[1]
            t_z = dist * math.sin(elevation)                       + self.auv_state[2]

            t_pos     = np.array([t_x, t_y, t_z])
            direction = self.auv_state[0:3] - t_pos
            t_vel     = (direction / np.linalg.norm(direction)) * self.torpedo_speed

            self.torpedoes.append({
                'id': i, 'pos': t_pos, 'vel': t_vel, 'active': True,
                'prev_xi': None, 'target_decoy_id': None,
            })

        # Initial minimum distance — fixed denominator for R_dist and R_survive
        active_torpedoes = [t for t in self.torpedoes if t['active']]
        if active_torpedoes:
            self.prev_min_distance = min(
                np.linalg.norm(self.auv_state[0:3] - t['pos'])
                for t in active_torpedoes
            )
        else:
            self.prev_min_distance = 0.0

        self.initial_min_distance = self.prev_min_distance
        self.current_min_distance = self.prev_min_distance
        # Reset unpenalised reward memory for hit-penalty computation
        self.prev_reward = 0.0

        return self._get_observation(), {}

    # ------------------------------------------------------------------
    # Single environment step
    # ------------------------------------------------------------------
    def step(self, action):
        discrete_action, cont_params = action

        target_vel_norm = (cont_params[0] + 1.0) / 2.0
        target_vel      = target_vel_norm * (self.MAX_SPEED - self.MIN_SPEED) + self.MIN_SPEED
        target_heading  = cont_params[1] * math.pi

        # ===== 1. AUV Kinematics =====
        current_vel   = self.auv_state[3]
        current_psi   = self.auv_state[8]
        current_theta = self.auv_state[7]   # pitch is at idx 7

        accel_u  = np.clip(0.5 * (target_vel - current_vel), -self.MAX_ACCEL, self.MAX_ACCEL)
        diff_psi = (target_heading - current_psi + math.pi) % (2 * math.pi) - math.pi
        r_cmd    = np.clip(0.5 * diff_psi, -self.MAX_YAW_RATE, self.MAX_YAW_RATE)

        # Update speed and yaw
        self.auv_state[3]  = np.clip(self.auv_state[3] + accel_u * self.dt,
                                     self.MIN_SPEED, self.MAX_SPEED)
        self.auv_state[8]  = (self.auv_state[8] + r_cmd * self.dt + math.pi) % (2 * math.pi) - math.pi
        self.auv_state[11] = r_cmd  # angular velocity r (yaw rate)

        # Position update
        x_dot = self.auv_state[3] * math.cos(current_theta) * math.cos(self.auv_state[8])
        y_dot = self.auv_state[3] * math.cos(current_theta) * math.sin(self.auv_state[8])
        z_dot = -self.auv_state[3] * math.sin(current_theta)

        self.auv_state[0] += x_dot * self.dt
        self.auv_state[1] += y_dot * self.dt
        self.auv_state[2]  = np.clip(self.auv_state[2] + z_dot * self.dt,
                                     0.0, self.MAX_DEPTH)

        auv_pos = self.auv_state[0:3]
        auv_vel = np.array([x_dot, y_dot, z_dot])

        # ===== 2. Decoy Deployment =====
        decoy_error       = False
        decoy_errors_step = 0
        for d_action in discrete_action:
            d_action = int(d_action)
            if d_action == 0:
                continue                                    # no launch
            elif 13 <= d_action <= 36:                      # hovering decoy
                if self.ammo_hovering > 0:
                    self.ammo_hovering    -= 1
                    self.decoy_id_counter += 1
                    self.active_decoys.append({
                        'id': self.decoy_id_counter, 'type': 'hovering',
                        'pos': auv_pos.copy(), 'vel': np.zeros(3),
                        'life': self.DECOY_LIFE_STEPS,
                    })
                else:
                    decoy_error        = True
                    decoy_errors_step += 1
            elif 1 <= d_action <= 12:                       # mobile decoy
                angle_idx = d_action % 6
                if self.ammo_mobile[angle_idx] > 0:
                    self.ammo_mobile[angle_idx] -= 1
                    self.decoy_id_counter       += 1
                    pre_designated_angle  = self.mobile_angles[angle_idx]
                    launch_heading        = self.auv_state[8] + np.deg2rad(pre_designated_angle)
                    decoy_v = np.array([
                        self.auv_state[3] * math.cos(launch_heading),
                        self.auv_state[3] * math.sin(launch_heading),
                        0.0,
                    ])
                    self.active_decoys.append({
                        'id': self.decoy_id_counter, 'type': 'mobile',
                        'pos': auv_pos.copy(), 'vel': decoy_v,
                        'life': self.DECOY_LIFE_STEPS,
                    })
                else:
                    decoy_error        = True
                    decoy_errors_step += 1

        # Advance / age decoys
        alive_decoys = []
        for d in self.active_decoys:
            if d['type'] == 'mobile':
                d['pos'] += d['vel'] * self.dt
            d['life'] -= 1
            if d['life'] > 0:
                alive_decoys.append(d)
        self.active_decoys = alive_decoys

        # ===== 3. Torpedo seeker, seduction, kinematics =====
        hit_auv             = False
        torpedoes_destroyed = 0

        for torp in self.torpedoes:
            if not torp['active']:
                continue

            torp_heading_vec = torp['vel'] / (np.linalg.norm(torp['vel']) + 1e-6)
            visible_targets  = []

            # AUV visible to seeker?
            vec_to_auv  = auv_pos - torp['pos']
            dist_to_auv = np.linalg.norm(vec_to_auv)
            if dist_to_auv < self.seeker_range:
                auv_dir      = vec_to_auv / (dist_to_auv + 1e-6)
                angle_to_auv = np.arccos(np.clip(np.dot(torp_heading_vec, auv_dir), -1.0, 1.0))
                if angle_to_auv <= self.seeker_fov / 2.0:
                    intensity = 1.0 / (dist_to_auv ** 2 + 1e-6)
                    visible_targets.append({
                        'type': 'auv', 'pos': auv_pos, 'vel': auv_vel,
                        'intensity': intensity,
                    })

            # Decoys visible to seeker?
            for decoy in self.active_decoys:
                vec_to_decoy  = decoy['pos'] - torp['pos']
                dist_to_decoy = np.linalg.norm(vec_to_decoy)
                if dist_to_decoy < self.seeker_range:
                    decoy_dir      = vec_to_decoy / (dist_to_decoy + 1e-6)
                    angle_to_decoy = np.arccos(np.clip(np.dot(torp_heading_vec, decoy_dir), -1.0, 1.0))
                    if angle_to_decoy <= self.seeker_fov / 2.0:
                        intensity = (self.DECOY_ACOUSTIC_MULTIPLIER /
                                     (dist_to_decoy ** 2 + 1e-6))
                        visible_targets.append({
                            'type': 'decoy', 'id': decoy['id'],
                            'pos': decoy['pos'], 'vel': decoy['vel'],
                            'intensity': intensity,
                        })

            # Pick highest-intensity target
            target_pos       = None
            target_vel_local = None
            if visible_targets:
                best_target      = max(visible_targets, key=lambda t: t['intensity'])
                target_pos       = best_target['pos']
                target_vel_local = best_target['vel']

                # Decoy hit?
                if best_target['type'] == 'decoy':
                    if np.linalg.norm(torp['pos'] - best_target['pos']) < self.hit_distance:
                        torp['active']     = False
                        target_id          = best_target['id']
                        self.active_decoys = [d for d in self.active_decoys if d['id'] != target_id]
                        if dist_to_auv < (self.hit_distance * 2.0):
                            hit_auv = True       # decoy detonated too close to AUV
                        else:
                            torpedoes_destroyed += 1
                        continue

            # Direct AUV hit?
            if dist_to_auv < self.hit_distance:
                hit_auv        = True
                torp['active'] = False
                continue

            # Pursuit kinematics (proportional navigation)
            if torp['active'] and target_pos is not None:
                rel_pos        = target_pos      - torp['pos']
                rel_vel        = target_vel_local - torp['vel']
                dist_to_target = np.linalg.norm(rel_pos)
                if dist_to_target > 0:
                    los_rate  = np.cross(rel_pos, rel_vel) / (dist_to_target ** 2)
                    v_closing = -np.dot(rel_vel, rel_pos / dist_to_target)
                    if v_closing > 0:
                        accel_dir = np.cross(los_rate, rel_pos / dist_to_target)
                        accel_cmd = self.N_gain * v_closing * accel_dir
                        torp['vel'] += accel_cmd * self.dt
                        torp['vel']  = (torp['vel'] / np.linalg.norm(torp['vel'])) * self.torpedo_speed

            torp['pos'] += torp['vel'] * self.dt

        # ===== 4. Termination / truncation =====
        self.current_step += 1
        all_destroyed = all(not t['active'] for t in self.torpedoes)
        termination   = hit_auv or all_destroyed
        truncation    = self.current_step >= self.max_steps

        # Update current minimum distance.
        active_torpedoes = [t for t in self.torpedoes if t['active']]
        if active_torpedoes:
            self.current_min_distance = min(
                np.linalg.norm(self.auv_state[0:3] - t['pos'])
                for t in active_torpedoes
            )
        else:
            self.current_min_distance = self.initial_min_distance

        # Compute per-step reward using new formulation.
        reward = self._compute_reward(hit_auv, torpedoes_destroyed)

        # Update prev_min_distance for next step
        self.prev_min_distance = self.current_min_distance
        self.prev_reward = self._last_unpenalised

        info = {
            "hit":             hit_auv,
            "won":             all_destroyed,
            "survived":        not hit_auv,
            "torps_destroyed": torpedoes_destroyed,
            "decoy_errors":    decoy_errors_step,   # logged but not penalised
        }
        return self._get_observation(), reward, termination, truncation, info

    # ------------------------------------------------------------------
    # Build observation: (n_torpedoes, 20) - AUV state || torpedo i state
    # ------------------------------------------------------------------
    def _get_observation(self):
        x, y, z         = self.auv_state[0:3]
        u, v_b, w       = self.auv_state[3:6]
        phi, theta, psi = self.auv_state[6:9]
        p, q, r         = self.auv_state[9:12]

        v_E = np.linalg.norm([u, v_b, w])

        obs_auv = [
            x / 5000.0,                 y / 5000.0,                 z / self.MAX_DEPTH,
            v_E / self.MAX_SPEED,       v_b / self.MAX_SPEED,       u / self.MAX_SPEED,
            z / self.MAX_DEPTH,         self.MAX_DEPTH / 1000.0,
            phi / math.pi,              theta / math.pi,            psi / (2 * math.pi),
            p,                           q,                          r / self.MAX_YAW_RATE,
            sum(self.ammo_mobile) / float(self.MAX_MOBILE_DECOYS),
            self.ammo_hovering / float(self.MAX_HOVERING_DECOYS),
        ]

        multi_obs = []
        for torp in self.torpedoes:
            if not torp['active']:
                torp_obs = [0.0, 0.0, 0.0, 1.0]
                multi_obs.append(obs_auv + torp_obs)
                continue

            rel_pos   = torp['pos'] - self.auv_state[0:3]
            true_dist = np.linalg.norm(rel_pos)
            l         = true_dist * (1.0 + rng.normal(0, self.noise_level))

            angle_to_torp = math.atan2(rel_pos[1], rel_pos[0])
            xi   = (angle_to_torp - psi + math.pi) % (2 * math.pi) - math.pi
            zeta = math.atan2(rel_pos[2], math.sqrt(rel_pos[0]**2 + rel_pos[1]**2 + 1e-6))

            if torp['prev_xi'] is None:
                omega = 0.0
            else:
                omega = ((xi - torp['prev_xi'] + math.pi) % (2 * math.pi) - math.pi) / self.dt
            torp['prev_xi'] = xi

            torp_obs = [xi / math.pi, zeta / (math.pi / 2), omega / math.pi, l / 5000.0]
            multi_obs.append(obs_auv + torp_obs)

        return np.array(multi_obs, dtype=np.float32)

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------
    def _compute_reward(self, hit_auv: bool, torpedoes_destroyed: int) -> float:
        """
        Per-step reward:

            R_t = ε₁·R_dist + ε₂·R_survive + R_decoy_bonus + R_hit_penalty

        Terms
        -----
        R_dist       (Eq. 11): ratio of current to initial minimum torpedo
                     distance. ≥1 means the AUV has opened or held the gap;
                     <1 means torpedoes have closed in. Bounded and finite
                     (goes to 0 only when a torpedo is at the AUV).

        R_survive    (Eq. 12): constant per-step survival credit that scales
                     with initial geometry — closer spawns reward harder
                     evasions more. Does NOT depend on current distance so it
                     cannot explode when a torpedo gets close.

        R_decoy_bonus: positive credit awarded immediately when a torpedo is
                     destroyed by a decoy this step. Encourages active use of
                     countermeasures.

        R_hit_penalty: large negative signal at the terminal step when the AUV
                     is struck. Scales off the previous unpenalised step reward
                     (self.prev_reward) via SCALING_FACTOR so the magnitude
                     adapts to the reward scale already seen in the episode.

        Parameters
        ----------
        hit_auv            : True if the AUV was struck by a torpedo this step.
        torpedoes_destroyed: Number of torpedoes destroyed by decoys this step.
        """
        eps1           = EPSILON1
        eps2           = EPSILON2
        scaling_factor = SCALING_FACTOR

        # Fixed denominator: initial closest torpedo distance, set once at reset.
        denom_init = self.initial_min_distance + 1e-6

        # (1) Distance reward: ratio of current to initial min distance.
        #     ≥ 1  → AUV has gained or maintained separation
        #     < 1  → torpedoes have closed in
        #     = 0  → all torpedoes destroyed (handled in step(): set to d₀ → ratio = 1)
        R_dist = self.current_min_distance / denom_init

        # (2) Survival reward: constant per step, scaled by initial geometry.
        #     Sums to (max_steps · torpedo_speed / d₀) over a full episode,
        #     which is larger for close spawns (harder scenarios reward more).
        R_survive = (self.max_steps * self.torpedo_speed) / denom_init

        # (3) Decoy bonus: immediate positive reward per torpedo destroyed.
        R_decoy_bonus = DECOY_REWARD * torpedoes_destroyed

        # (4) Hit penalty: large negative signal at the hit step only.
        #     Scales off self.prev_reward (the unpenalised reward from the
        #     previous step) so the penalty magnitude is proportional to what
        #     the agent was earning — a meaningful deterrent regardless of
        #     reward scale.
        R_hit_penalty = 0.0
        if hit_auv:
            R_hit_penalty = -scaling_factor * self.prev_reward

        self._last_unpenalised = (eps1 * R_dist) + (eps2 * R_survive) + R_decoy_bonus

        R_t = self._last_unpenalised + R_hit_penalty
        return R_t
