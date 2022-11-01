# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import torch

from isaacgym import gymutil, gymtorch, gymapi, torch_utils
from .base.vec_task import VecTask

from isaacgymenvs.utils.torch_jit_utils import *

class Twip(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.reset_dist = self.cfg["env"]["resetDist"]

        self.max_push_effort = self.cfg["env"]["maxEffort"]
        self.max_episode_length = 200

        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]

        # Observations:
        # 0:13 - root state
        # 13:29 - DOF states
        num_obs = 5

        # Actions:
        # 0:8 - rotor DOF position targets
        # 8:12 - rotor thrust magnitudes
        num_acts = 2

        self.cfg["env"]["numObservations"] = num_obs
        self.cfg["env"]["numActions"] = num_acts

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        print("dof state", self.dof_state)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)


        self.all_actor_indices = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)

        # Actors x Info
        #self.root_positions = self.root_states[:, 0:3]
        self.root_orientations = self.root_states[:, 3:7]
        print("get_euler_xyz(self.root_states[:, 3:7])", get_euler_xyz(self.root_states[:, 3:7]))
        print("self.root_states[:, 3:7]", self.root_states[:, 3:7])
        self.root_pitch = get_euler_xyz(self.root_states[:, 3:7])[1]
        self.root_yaw = get_euler_xyz(self.root_states[:, 3:7])[2]

        self.root_x_d = quat_rotate_inverse(self.root_orientations, self.root_states[:, 7:10])[:, :2]    #
        #print("######", self.root_x_d[:, 0])
        #self.root_pitch = torch_utils.get_euler_xyz(self.root_states[:, 3:7])[1]
        self.root_angular_vels = self.root_states[:, 10:13] #pitch_d, yaw_d

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)


        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:, 2] = 0.35

        self.initial_dof_states = self.dof_state.clone()

        self.commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_y = self.commands.view(self.num_envs, 3)[..., 1]
        self.commands_x = self.commands.view(self.num_envs, 3)[..., 0]
        self.commands_yaw = self.commands.view(self.num_envs, 3)[..., 2]

    def create_sim(self):
        # set the up axis to be z-up given that assets are y-up by default
        self.up_axis = self.cfg["sim"]["up_axis"]

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        # set the normal force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) if self.up_axis == 'z' else gymapi.Vec3(0.0, 1.0, 0.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        # define plane on which environments are initialized
        lower = gymapi.Vec3(0.5 * -spacing, -spacing, 0.0) if self.up_axis == 'z' else gymapi.Vec3(0.5 * -spacing, 0.0, -spacing)
        upper = gymapi.Vec3(0.5 * spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        asset_file = "urdf/twip.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        twip_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(twip_asset)

        pose = gymapi.Transform()
        if self.up_axis == 'z':
            pose.p.z = 0.35
            # asset is rotated z-up by default, no additional rotations needed
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        else:
            pose.p.y = 2.0
            pose.r = gymapi.Quat(-np.sqrt(2)/2, 0.0, 0.0, np.sqrt(2)/2)

        self.twip_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            twip_handle = self.gym.create_actor(env_ptr, twip_asset, pose, "twip", i, 1, 0)

            dof_props = self.gym.get_actor_dof_properties(env_ptr, twip_handle)
            dof_props['driveMode'][0] = gymapi.DOF_MODE_EFFORT
            dof_props['driveMode'][1] = gymapi.DOF_MODE_EFFORT
            dof_props['stiffness'][:] = 0.0
            dof_props['damping'][:] = 0.0
            self.gym.set_actor_dof_properties(env_ptr, twip_handle, dof_props)

            self.envs.append(env_ptr)
            self.twip_handles.append(twip_handle)

    def compute_reward(self):
        # retrieve environment observations from buffer
        twip_pitch_d = self.obs_buf[:, 3]
        twip_yaw_d = self.obs_buf[:, 4]
        twip_vel = self.obs_buf[:, :2]

        twip_pitch = self.obs_buf[:, 2]

        self.rew_buf[:], self.reset_buf[:] = compute_twip_reward(
            twip_vel, twip_pitch, twip_pitch_d, twip_yaw_d,
            self.commands, self.reset_buf, self.progress_buf, self.max_episode_length
        )


    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)



        #self.obs_buf[env_ids, 0] = self.root_x_d[env_ids].squeeze()
        self.obs_buf[env_ids, 0] = quat_rotate_inverse(self.root_states[:, 3:7], self.root_states[:, 7:10])[env_ids,0].squeeze() #x_d
        self.obs_buf[env_ids, 1] = quat_rotate_inverse(self.root_states[:, 3:7], self.root_states[:, 7:10])[env_ids,1].squeeze() #y_d
        self.obs_buf[env_ids, 2] = get_euler_xyz(self.root_states[env_ids, 3:7])[1].squeeze() #pitch
        
        #print("obs pitch pre rot", self.obs_buf[env_ids, 2])
        num_resets = len(env_ids)
        #env_ids_int32 = env_ids.to(dtype=torch.int32)
        treshold = torch.tensor([2], device=self.device)
        pi_tensor = torch.tensor([6.28318], device=self.device)
        self.obs_buf[env_ids, 2] = torch.where(self.obs_buf[env_ids, 2] > treshold, self.obs_buf[env_ids, 2] - pi_tensor, self.obs_buf[env_ids, 2])
        self.obs_buf[env_ids, 2] = torch.where(self.obs_buf[env_ids, 2] < -treshold, self.obs_buf[env_ids, 2] + pi_tensor, self.obs_buf[env_ids, 2])
        #self.obs_buf[env_ids, 3] = get_euler_xyz(self.root_states[:, 3:7])[env_ids,2].squeeze() #yaw
        #self.obs_buf[env_ids, 4] = self.root_orientations[env_ids,2].squeeze()
        #self.obs_buf[env_ids, 5] = self.root_orientations[env_ids,3].squeeze()
        self.obs_buf[env_ids, 3] = self.root_states[env_ids, 11].squeeze() #pitch_d
        self.obs_buf[env_ids, 4] = self.root_states[env_ids, 12].squeeze() #yaw_d


        '''print("obs pitch", self.obs_buf[env_ids, 2])
        print("obs pitch_d", self.obs_buf[env_ids, 3])
        print("obs yaw_d", self.obs_buf[env_ids, 4])'''

        
        return self.obs_buf

    def reset_idx(self, env_ids):
        print("################ RESET")

        num_resets = len(env_ids)
        
        positions = 0.2 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5)
        velocities = 0.5 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5)

        #self.dof_pos[env_ids, :] = positions[:]
        #self.dof_vel[env_ids, :] = velocities[:]

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.root_states[env_ids] = self.initial_root_states[env_ids]
        temp_euler = get_euler_xyz(self.root_states[env_ids, 3:7])
        treshold = torch.tensor([2], device=self.device)
        pi_tensor = torch.tensor([6.28318], device=self.device)
        #minus_pi_tensor = torch.tensor([-3.14], (num_resets, 1), device=self.device)

        pitch = torch.tensor(temp_euler[1])
        yaw = torch.tensor(temp_euler[2])
        
        pitch = torch.where(pitch > treshold, pitch - pi_tensor, pitch)
        pitch = torch.where(pitch < -treshold, pitch + pi_tensor, pitch)
        
        '''print("temp_euler", temp_euler)
        print("temp_euler_pitch", temp_euler[1])
        print("tensor temp_euler_pitch", torch.tensor(temp_euler[1]))
        print("torch_rand_float(-0.3, 0.3, (num_resets, 1), self.device).flatten()", torch_rand_float(-0.3, 0.3, (num_resets, 1), self.device).flatten())
        print("self.root_states[env_ids, 3:7]", self.root_states[env_ids, 3:7])'''
        
        pitch += pitch + torch_rand_float(-0.4, 0.4, (num_resets, 1), self.device).flatten() #randomized pitch
        yaw = torch.tensor(temp_euler[2]) + torch_rand_float(-3, 3, (num_resets, 1), self.device).flatten() #randomized yaw
        #temp_euler[1] += torch_rand_float(-0.3, 0.3, (num_resets, 1), self.device).flatten() #randomized pitch
        #temp_euler[2] += torch_rand_float(-1, 1, (num_resets, 1), self.device).flatten() #randomized yaw
        
        self.root_states[env_ids, 3:7] = quat_from_euler_xyz(temp_euler[0], pitch, yaw)
        #self.root_states[env_ids, 4] += torch_rand_float(-0.2, 0.1, (num_resets, 1), self.device).flatten()
        #self.root_states[env_ids, 1] += torch_rand_float(-1.5, 1.5, (num_resets, 1), self.device).flatten()

   

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.initial_dof_states),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))



        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        self.commands_x[env_ids] = torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze()
        self.commands_y[env_ids] = torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze()
        self.commands_yaw[env_ids] = torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze()



    def pre_physics_step(self, actions):
        
        #actions_tensor = torch.zeros(self.num_envs * self.num_dof, device=self.device, dtype=torch.float) * self.max_push_effort
        #[::self.num_dof] = actions.to(self.device).squeeze() * self.max_push_effort
        actions = actions.to(self.device)
        torques = gymtorch.unwrap_tensor(actions* self.max_push_effort)
        self.gym.set_dof_actuation_force_tensor(self.sim, torques)

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        self.compute_observations()
        self.compute_reward()

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_twip_reward(twip_vel, twip_pitch, twip_pitch_d, twip_yaw_d,
                            commands, reset_buf, progress_buf, max_episode_length):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    #print("commands", commands[:,:2])
    #reward = -torch.abs(twip_vel[:, 0] - 0) - torch.abs(twip_vel[:, 1] - 0)
    #reward = -torch.sum(torch.square(commands[:, :2] - twip_vel[:, :2]), dim=1)
    
    reward = -(torch.abs(twip_pitch) - 0)*1
    reward += -(torch.abs(twip_pitch_d) - 0)*0.3 - (torch.abs(twip_yaw_d) - 0)*0.3


    # adjust reward for reset agents
    #print("twip_pitch", twip_pitch)
    reward = torch.where(torch.abs(twip_pitch) > np.pi / 3, torch.ones_like(reward) * -10.0, reward)
    
    reset = torch.where(torch.abs(twip_pitch) > np.pi / 3 , torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)

    #reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)



    return reward, reset
