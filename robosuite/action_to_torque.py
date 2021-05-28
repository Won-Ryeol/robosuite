import numpy as np
import scipy
from enum import Enum

import robosuite.utils.transform_utils as T

#from baselines.baselines import logger
import logging
logger = logging.getLogger(__name__)

from scipy.interpolate import CubicSpline

class Controller():
    def __init__(self,
                 control_max,
                 control_min,
                 max_action,
                 min_action,
                 control_freq=20,
                 impedance_flag=False,
                 kp_max=None,
                 kp_min=None,
                 damping_max=None,
                 damping_min=None,
                 initial_joint=None,
                 position_limits=[[0, 0, 0], [0, 0, 0]],
                 orientation_limits=[[0, 0, 0], [0, 0, 0]],
                 interpolation=None,
                 **kwargs
                 ):

        # If the action includes impedance parameters
        self.impedance_flag = impedance_flag

        # Initial joint configuration we use for the task in the null space
        self.initial_joint = initial_joint

        # Upper and lower limits to the input action (only pos/ori)
        self.control_max = control_max
        self.control_min = control_min

        # Dimensionality of the action
        self.control_dim = self.control_max.shape[0]

        if self.impedance_flag:
            impedance_max = np.hstack((kp_max, damping_max))
            impedance_min = np.hstack((kp_min, damping_min))
            self.control_max = np.hstack((self.control_max, impedance_max))
            self.control_min = np.hstack((self.control_min, impedance_min))

        # Limits to the policy outputs
        self.input_max = max_action
        self.input_min = min_action

        # This handles when the mean of max and min control is not zero -> actions are around that mean
        self.action_scale = abs(self.control_max - self.control_min) / abs(max_action - min_action)
        self.action_output_transform = (self.control_max + self.control_min) / 2.0
        self.action_input_transform = (max_action + min_action) / 2.0

        self.control_freq = control_freq  # control steps per second

        self.interpolation = interpolation

        self.ramp_ratio = 0.20  # Percentage of the time between policy timesteps used for interpolation

        self.position_limits = position_limits
        self.orientation_limits = orientation_limits

        # Initialize the remaining attributes
        self.model_timestep = None
        self.interpolation_steps = None
        self.current_position = None
        self.current_orientation_mat = None
        self.current_lin_velocity = None
        self.current_ang_velocity = None
        self.current_joint_position = None
        self.current_joint_velocity = None
        self.Jx = None
        self.Jr = None
        self.J_full = None

    def reset(self):
        """
        Resets the internal values of the controller
        """
        pass

    def transform_action(self, action):
        """
        Scale the action to go to the right min and max
        """
        action = np.clip(action, self.input_min, self.input_max)
        transformed_action = (action - self.action_input_transform) * self.action_scale + self.action_output_transform

        return transformed_action

    def update_model(self):
        """
        Updates the state of the robot used to compute the control command
        """
        inter_steps = 2000.0
        model_timestep = 0.002
        c_pos = np.array([0.44498526729600935, 0.0, 1.1077636435061193])
        c_ori_mat = np.array([[ 9.91914615e-01, -4.88641358e-04,  1.26906102e-01],
                              [-4.92624356e-04, -9.99999879e-01,  2.66307159e-18],
                              [ 1.26906086e-01, -6.25170366e-05, -9.91914735e-01]])
        c_lin_vel = np.zeros(3)
        c_ang_vel = np.zeros(3)
        c_joint_pos = np.array([ 0.        ,  0.19634954,  0.        , -2.61799388,  0.        ,  2.94159265,  -0.78539816])       
        c_joint_vel = np.zeros(7)
        J_x = np.array([[ 0.        , -0.13823636,  0.        ,  0.43206955,  0.        ,  0.09496714,  0.        ],
                        [ 0.44498527,  0.        ,  0.46340358,  0.        , -0.06498824,  0.        ,  0.        ],
                        [ 0.        , -0.44498527,  0.        ,  0.30242194,  0.        ,  0.10086745,  0.        ]])
        J_r = np.array([[ 0.        ,  0.        ,  0.19509032,  0.        ,  0.32143947,  0.        ,  0.1269061 ],
                        [ 0.        ,  1.        ,  0.        , -1.        ,  0.        ,  -1.       ,  0.        ],
                        [ 1.        ,  0.        ,  0.98078528,  0.        , -0.94693013,  0.        , -0.99191473]])
        
        self.J_full = np.vstack([self.Jx, self.Jr])

    def update_mass_matrix(self):
        """
        Update the mass matrix.
        sim - Mujoco simulation object
        joint_index - list of joint position indices in Mujoco
        """
        # mass_matrix = np.ndarray(shape=(len(sim.data.qvel) ** 2,), dtype=np.float64, order='C')
        # mujoco_py.cymj._mj_fullM(sim.model, mass_matrix, sim.data.qM)
        # mass_matrix = np.reshape(mass_matrix, (len(sim.data.qvel), len(sim.data.qvel)))
        # self.mass_matrix = mass_matrix[joint_index, :][:, joint_index]
        self.mass_matrix = np.array([[ 2.25726539e+00, -2.62750718e-06,  1.60412060e+00,
                                        2.62750718e-06, -5.43221706e-01,  2.62750718e-06,
                                        -1.98486117e-01],
                                    [-2.62750718e-06,  2.28637709e+00, -6.58357876e-06,
                                        -1.16381227e+00, -4.11331754e-06, -4.00056093e-01,
                                        2.53844900e-18],
                                    [ 1.60412060e+00, -6.58357876e-06,  1.60070894e+00,
                                        6.58357876e-06, -5.02987276e-01,  6.58357876e-06,
                                        -1.89718057e-01],
                                    [ 2.62750718e-06, -1.16381227e+00,  6.58357876e-06,
                                        1.43272976e+00,  4.11331754e-06,  4.01913080e-01,
                                        -2.53381454e-18],
                                    [-5.43221706e-01, -4.11331754e-06, -5.02987276e-01,
                                        4.11331754e-06,  5.11565111e-01,  4.11331754e-06,
                                        1.96115254e-01],
                                    [ 2.62750718e-06, -4.00056093e-01,  6.58357876e-06,
                                        4.01913080e-01,  4.11331754e-06,  3.22014223e-01,
                                        -2.53991699e-18],
                                    [-1.98486117e-01,  2.53844900e-18, -1.89718057e-01,
                                        -2.53381454e-18,  1.96115254e-01, -2.53991699e-18,
                                        2.00104011e-01]])

    def set_goal_impedance(self, action):
        """
        Interpret the action as the intended impedance. The impedance is not set
        directly in case interpolation is enabled.
        """
        if self.use_delta_impedance:
            # clip resulting kp and damping
            self.goal_kp = np.clip(self.impedance_kp[self.action_mask] + action[self.kp_index[0]:self.kp_index[1]],
                                   self.kp_min, self.kp_max)
            self.goal_damping = np.clip(
                self.impedance_damping[self.action_mask] + action[self.damping_index[0]:self.damping_index[1]], self.damping_min,
                self.damping_max)
        else:
            # no clipped is needed here, since the action has already been scaled
            self.goal_kp = action[self.kp_index[0]:self.kp_index[1]]
            self.goal_damping = action[self.damping_index[0]:self.damping_index[1]]

    def linear_interpolate(self, last_goal, goal):
        """
        Set self.linear to be a function interpolating between last_goal and goal based on the ramp_ratio
        """
        # We interpolate to reach the commanded desired position in self.ramp_ratio % of the time we have this goal
        delta_x_per_step = (goal - last_goal) / self.interpolation_steps
        self.linear = np.array(
            [(last_goal + i * delta_x_per_step) for i in range(1, int(self.interpolation_steps) + 1)])

    def interpolate_impedance(self, starting_kp, starting_damping, goal_kp, goal_damping):
        """
        Set self.update_impedance to be a function for generating the impedance given the timestep
        """
        delta_kp_per_step = (goal_kp - starting_kp[self.action_mask]) / self.interpolation_steps
        delta_damping_per_step = (goal_damping - starting_damping[self.action_mask]) / self.interpolation_steps

        def update_impedance(index):
            if index < self.interpolation_steps - 1:
                self.impedance_kp[self.action_mask] += delta_kp_per_step
                self.impedance_damping[self.action_mask] += delta_damping_per_step

        self.update_impedance = update_impedance

    def calculate_orientation_error(self, desired, current):
        """
        Optimized function to determine orientation error
        """

        def cross_product(vec1, vec2):
            S = np.array(([0, -vec1[2], vec1[1]],
                          [vec1[2], 0, -vec1[0]],
                          [-vec1[1], vec1[0], 0]))

            return np.dot(S, vec2)

        rc1 = current[0:3, 0]
        rc2 = current[0:3, 1]
        rc3 = current[0:3, 2]
        rd1 = desired[0:3, 0]
        rd2 = desired[0:3, 1]
        rd3 = desired[0:3, 2]

        orientation_error = 0.5 * (cross_product(rc1, rd1) + cross_product(rc2, rd2) + cross_product(rc3, rd3))

        return orientation_error

    def action_to_torques(self, action, policy_step):
        raise NotImplementedError

    @property
    def action_dim(self):
        """
        Returns dimensionality of the actions
        """
        dim = self.control_dim
        if self.impedance_flag:
            # Includes (stacked) state vector, kp vector, and damping vector
            dim = dim * 3
        return dim

    @property
    def kp_index(self):
        """
        Indices of the kp values in the action vector
        """
        start_index = self.control_dim
        end_index = start_index + self.control_dim

        if self.impedance_flag:
            return (start_index, end_index)
        else:
            return None

    @property
    def damping_index(self):
        """
        Indices of the damping ratio values in the action vector
        """
        start_index = self.kp_index[1]
        end_index = start_index + self.control_dim

        if self.impedance_flag:
            return (start_index, end_index)
        else:
            return None

    @property
    def action_mask(self):
        raise NotImplementedError
    
class PositionOrientationController(Controller):
    """
    Class to interpret actions as cartesian desired position and orientation (and impedance values)
    """

    def __init__(self,
                 control_range_pos,
                 control_range_ori,
                 kp_max,
                 kp_max_abs_delta,
                 kp_min,
                 damping_max,
                 damping_max_abs_delta,
                 damping_min,
                 use_delta_impedance,
                 initial_impedance_pos,
                 initial_impedance_ori,
                 initial_damping,
                 initial_joint=None,
                 control_freq=20,
                 max_action=1,
                 min_action=-1,
                 impedance_flag=False,
                 position_limits=[[0, 0, 0], [0, 0, 0]],
                 orientation_limits=[[0, 0, 0], [0, 0, 0]],
                 interpolation=None,
                 **kwargs
                 ):
        control_max = np.ones(3) * control_range_pos
        if control_range_ori is not None:
            control_max = np.concatenate([control_max, np.ones(3) * control_range_ori])
        control_min = -1 * control_max
        kp_max = (np.ones(6) * kp_max)[self.action_mask]
        kp_max_abs_delta = (np.ones(6) * kp_max_abs_delta)[self.action_mask]
        kp_min = (np.ones(6) * kp_min)[self.action_mask]
        damping_max = (np.ones(6) * damping_max)[self.action_mask]
        damping_max_abs_delta = (np.ones(6) * damping_max_abs_delta)[self.action_mask]
        damping_min = (np.ones(6) * damping_min)[self.action_mask]
        initial_impedance = np.concatenate([np.ones(3) * initial_impedance_pos, np.ones(3) * initial_impedance_ori])
        initial_damping = np.ones(6) * initial_damping

        self.use_delta_impedance = use_delta_impedance

        if self.use_delta_impedance:
            # provide range of possible delta impedances
            kp_param_max = kp_max_abs_delta
            kp_param_min = -kp_max_abs_delta
            damping_param_max = damping_max_abs_delta
            damping_param_min = -damping_max_abs_delta

            # store actual ranges for manual clipping
            self.kp_max = kp_max
            self.kp_min = kp_min
            self.damping_max = damping_max
            self.damping_min = damping_min
        else:
            # just use ranges directly
            kp_param_max = kp_max
            kp_param_min = kp_min
            damping_param_max = damping_max
            damping_param_min = damping_min

        super(PositionOrientationController, self).__init__(
            control_max=control_max,
            control_min=control_min,
            max_action=max_action,
            min_action=min_action,
            impedance_flag=impedance_flag,
            kp_max=kp_param_max,
            kp_min=kp_param_min,
            damping_max=damping_param_max,
            damping_min=damping_param_min,
            initial_joint=initial_joint,
            control_freq=control_freq,
            position_limits=position_limits,
            orientation_limits=orientation_limits,
            interpolation=interpolation,
            **kwargs
        )

        self.impedance_kp = np.array(initial_impedance).astype('float64')
        self.impedance_damping = np.array(initial_damping).astype('float64')

        self.step = 0
        self.interpolate = True

        self.last_goal_position = np.array((0, 0, 0))
        self.last_goal_orientation = np.eye(3)

    def reset(self):
        super().reset()
        self.step = 0
        self.last_goal_position = np.array((0, 0, 0))
        self.last_goal_orientation = np.eye(3)

    def interpolate_position(self, starting_position, last_goal_position, goal_position, current_vel):

        if self.interpolation == "cubic":
            # We interpolate to reach the commanded desired position in self.ramp_ratio % of the time we have this goal
            time = [0, self.interpolation_steps]
            position = np.vstack((starting_position, goal_position))
            self.spline_pos = CubicSpline(time, position, bc_type=((1, current_vel), (1, (0, 0, 0))), axis=0)
        elif self.interpolation == 'linear':
            delta_x_per_step = (goal_position - last_goal_position) / self.interpolation_steps
            self.linear_pos = np.array(
                [(last_goal_position + i * delta_x_per_step) for i in range(1, int(self.interpolation_steps) + 1)])
        elif self.interpolation == None:
            pass
        else:
            logger.error("[Controller] Invalid interpolation! Please specify 'cubic' or 'linear'.")
            exit(-1)

    def interpolate_orientation(self, starting_orientation, last_goal_orientation, goal_orientation, current_vel):
        # We interpolate to reach the commanded desired position in self.ramp_ratio % of the time we have this goal
        if self.interpolation == "cubic":
            time = [0, self.interpolation_steps]
            orientation_error = self.calculate_orientation_error(desired=goal_orientation, current=starting_orientation)
            orientation = np.vstack(([0, 0, 0], orientation_error))
            self.spline_ori = CubicSpline(time, orientation, bc_type=((1, current_vel), (1, (0, 0, 0))), axis=0)
            self.orientation_initial_interpolation = starting_orientation
        elif self.interpolation == 'linear':
            orientation_error = self.calculate_orientation_error(desired=goal_orientation,
                                                                 current=last_goal_orientation)
            delta_r_per_step = orientation_error / self.interpolation_steps
            self.linear_ori = np.array([i * delta_r_per_step for i in range(1, int(self.interpolation_steps) + 1)])
            self.orientation_initial_interpolation = last_goal_orientation
        elif self.interpolation == None:
            pass
        else:
            logger.error("[Controller] Invalid interpolation! Please specify 'cubic' or 'linear'.")
            exit(-1)

    def action_to_torques(self, action, policy_step):
        """
        Given the next action, output joint torques for the robot.
        Assumes the robot's model is updated.
        """
        action = self.transform_action(action)

        # This is computed only when we receive a new desired goal from policy
        
        if policy_step == True:
            self.step = 0
            self.set_goal_position(action)
            self.set_goal_orientation(action)
            if self.impedance_flag: self.set_goal_impedance(
                action)  # this takes into account whether or not it's delta impedance

            if self.interpolation:
                # The first time we interpolate we don't have a previous goal value -> We set it to the current robot position+orientation
                if np.linalg.norm(self.last_goal_position) == 0:
                    self.last_goal_position = self.current_position
                if (self.last_goal_orientation == np.eye(self.last_goal_orientation.shape[0])).all():
                    self.last_goal_orientation = self.current_orientation_mat
                # set goals for next round of interpolation - TODO rename these functions?
                self.interpolate_position(self.current_position, self.last_goal_position, self.goal_position,
                                          self.current_lin_velocity)
                self.interpolate_orientation(self.current_orientation_mat, self.last_goal_orientation,
                                             self.goal_orientation, self.current_ang_velocity)

            # handle impedances
            if self.impedance_flag:
                if self.interpolation:
                    # set goals for next round of interpolation
                    self.interpolate_impedance(self.impedance_kp, self.impedance_damping, self.goal_kp, self.goal_damping)
                else:
                    # update impedances immediately
                    self.impedance_kp[self.action_mask] = self.goal_kp
                    self.impedance_damping[self.action_mask] = self.goal_damping

        if self.interpolation:
            if self.interpolation == 'cubic':
                self.last_goal_position = self.spline_pos(self.step)
                goal_orientation_delta = self.spline_ori(self.step)
            elif self.interpolation == 'linear':
                self.last_goal_position = self.linear_pos[self.step]
                goal_orientation_delta = self.linear_ori[self.step]
            else:
                logger.error("[Controller] Invalid interpolation! Please specify 'cubic' or 'linear'.")
                exit(-1)

            if self.impedance_flag: self.update_impedance(self.step)

            self.last_goal_orientation = np.dot((T.euler2mat(-goal_orientation_delta).T),
                                                self.orientation_initial_interpolation)

            # After self.ramp_ratio % of the time we have reached the desired pose and stay constant
            if self.step < self.interpolation_steps - 1:
                self.step += 1
        else:
            self.last_goal_position = np.array((self.goal_position))
            self.last_goal_orientation = self.goal_orientation

            if self.impedance_flag:
                self.impedance_kp = action[self.kp_index[0]:self.kp_index[1]]
                self.impedance_damping = action[self.damping_index[0]:self.damping_index[1]]

        position_error = self.last_goal_position - self.current_position
        #print("Position err: {}".format(position_error))
        orientation_error = self.calculate_orientation_error(desired=self.last_goal_orientation,
                                                             current=self.current_orientation_mat)

        # always ensure critical damping TODO - technically this is called unneccessarily if the impedance_flag is not set
        self.impedance_kv = 2 * np.sqrt(self.impedance_kp) * self.impedance_damping

        return self.calculate_impedance_torques(position_error, orientation_error)

    def calculate_impedance_torques(self, position_error, orientation_error):
        """
        Given the current errors in position and orientation, return the desired torques per joint
        """
        desired_force = (np.multiply(np.array(position_error), np.array(self.impedance_kp[0:3]))
                         - np.multiply(np.array(self.current_lin_velocity), self.impedance_kv[0:3]))

        desired_torque = (np.multiply(np.array(orientation_error), np.array(self.impedance_kp[3:6]))
                          - np.multiply(np.array(self.current_ang_velocity), self.impedance_kv[3:6]))

        uncoupling = True
        if (uncoupling):
            decoupled_force = np.dot(self.lambda_x_matrix, desired_force)
            decoupled_torque = np.dot(self.lambda_r_matrix, desired_torque)
            decoupled_wrench = np.concatenate([decoupled_force, decoupled_torque])
        else:
            desired_wrench = np.concatenate([desired_force, desired_torque])
            decoupled_wrench = np.dot(self.lambda_matrix, desired_wrench)

        torques = np.dot(self.J_full.T, decoupled_wrench)

        if self.initial_joint is not None:
            # TODO where does 10 come from?
            joint_kp = 10
            joint_kv = np.sqrt(joint_kp) * 2
            pose_torques = np.dot(self.mass_matrix, (joint_kp * (
                        self.initial_joint - self.current_joint_position) - joint_kv * self.current_joint_velocity))
            nullspace_torques = np.dot(self.nullspace_matrix.transpose(), pose_torques)
            torques += nullspace_torques
            self.torques = torques

        return torques

    def update_model(self, joint_index):

        super().update_model()

        self.update_mass_matrix()
        self.update_model_opspace(joint_index)

    def update_model_opspace(self, joint_index):
        """
        Updates the following:
        -Lambda matrix (full, linear, and rotational)
        -Nullspace matrix

        joint_index - list of joint position indices in Mujoco
        """
        mass_matrix_inv = scipy.linalg.inv(self.mass_matrix)

        # J M^-1 J^T
        lambda_matrix_inv = np.dot(
            np.dot(self.J_full, mass_matrix_inv),
            self.J_full.transpose()
        )

        # (J M^-1 J^T)^-1
        self.lambda_matrix = scipy.linalg.inv(lambda_matrix_inv)

        # Jx M^-1 Jx^T
        lambda_x_matrix_inv = np.dot(
            np.dot(self.Jx, mass_matrix_inv),
            self.Jx.transpose()
        )

        # Jr M^-1 Jr^T
        lambda_r_matrix_inv = np.dot(
            np.dot(self.Jr, mass_matrix_inv),
            self.Jr.transpose()
        )

        # take the inverse, but zero out elements in cases of a singularity
        svd_u, svd_s, svd_v = np.linalg.svd(lambda_x_matrix_inv)
        singularity_threshold = 0.00025
        svd_s_inv = [0 if x < singularity_threshold else 1. / x for x in svd_s]
        self.lambda_x_matrix = svd_v.T.dot(np.diag(svd_s_inv)).dot(svd_u.T)

        svd_u, svd_s, svd_v = np.linalg.svd(lambda_r_matrix_inv)
        singularity_threshold = 0.00025
        svd_s_inv = [0 if x < singularity_threshold else 1. / x for x in svd_s]
        self.lambda_r_matrix = svd_v.T.dot(np.diag(svd_s_inv)).dot(svd_u.T)

        if self.initial_joint is not None:
            Jbar = np.dot(mass_matrix_inv, self.J_full.transpose()).dot(self.lambda_matrix)
            self.nullspace_matrix = np.eye(len(joint_index), len(joint_index)) - np.dot(Jbar, self.J_full)

    def set_goal_position(self, action, position=None):
        if position is not None:
            self._goal_position = position
        else:
            self._goal_position = self.current_position + action[0:3]
            if np.array(self.position_limits).any():
                for idx in range(3):
                    self._goal_position[idx] = np.clip(self._goal_position[idx], self.position_limits[0][idx],
                                                       self.position_limits[1][idx])

    def set_goal_orientation(self, action, orientation=None):
        if orientation is not None:
            self._goal_orientation = orientation
        else:
            rotation_mat_error = T.euler2mat(-action[3:6])
            self._goal_orientation = np.dot((rotation_mat_error).T, self.current_orientation_mat)
            if np.array(self.orientation_limits).any():
                # TODO: Limit rotation!
                euler = T.mat2euler(self._goal_orientation)

                limited = False
                for idx in range(3):
                    if self.orientation_limits[0][idx] < self.orientation_limits[1][idx]:  # Normal angle sector meaning
                        if euler[idx] > self.orientation_limits[0][idx] and euler[idx] < self.orientation_limits[1][
                            idx]:
                            continue
                        else:
                            limited = True
                            dist_to_lower = euler[idx] - self.orientation_limits[0][idx]
                            if dist_to_lower > np.pi:
                                dist_to_lower -= 2 * np.pi
                            elif dist_to_lower < -np.pi:
                                dist_to_lower += 2 * np.pi

                            dist_to_higher = euler[idx] - self.orientation_limits[1][idx]
                            if dist_to_lower > np.pi:
                                dist_to_higher -= 2 * np.pi
                            elif dist_to_lower < -np.pi:
                                dist_to_higher += 2 * np.pi

                            if dist_to_lower < dist_to_higher:
                                euler[idx] = self.orientation_limits[0][idx]
                            else:
                                euler[idx] = self.orientation_limits[1][idx]
                    else:  # Inverted angle sector meaning
                        if euler[idx] > self.orientation_limits[0][idx] or euler[idx] < self.orientation_limits[1][idx]:
                            continue
                        else:
                            limited = True
                            dist_to_lower = euler[idx] - self.orientation_limits[0][idx]
                            if dist_to_lower > np.pi:
                                dist_to_lower -= 2 * np.pi
                            elif dist_to_lower < -np.pi:
                                dist_to_lower += 2 * np.pi

                            dist_to_higher = euler[idx] - self.orientation_limits[1][idx]
                            if dist_to_lower > np.pi:
                                dist_to_higher -= 2 * np.pi
                            elif dist_to_lower < -np.pi:
                                dist_to_higher += 2 * np.pi

                            if dist_to_lower < dist_to_higher:
                                euler[idx] = self.orientation_limits[0][idx]
                            else:
                                euler[idx] = self.orientation_limits[1][idx]
                if limited:
                    self._goal_orientation = T.euler2mat(np.array([euler[1], euler[0], euler[2]]))

    @property
    def action_mask(self):
        # TODO - why can't this be control_dim like the others?
        return np.array((0, 1, 2, 3, 4, 5))

    # return np.arange(self.control_dim)

    @property
    def goal_orientation(self):
        return self._goal_orientation

    @property
    def goal_position(self):
        return self._goal_position

if __name__ == "__main__":
    test = PositionOrientationController(
                 control_range_pos = 0.05,
                 control_range_ori = 0.2,
                 kp_max = 300,
                 kp_max_abs_delta = 10,
                 kp_min = 10,
                 damping_max = 2,
                 damping_max_abs_delta = 0.1,
                 damping_min = 0,
                 use_delta_impedance = False,
                 initial_impedance_pos = 150,
                 initial_impedance_ori = 150,
                 initial_damping = 1,
                 initial_joint = None,
                 control_freq = 20,
                 max_action = 1,
                 min_action = -1,
                 impedance_flag = False,
                 position_limits = [[0, 0, 0], [0, 0, 0]],
                 orientation_limits = [[0, 0, 0], [0, 0, 0]],
                 interpolation = "linear"
                 )

    test_action = np.array([-0.2, 1.4, 0.5, -0.4, 0.9, -1.5])

    joint_index = [0, 1, 2, 3, 4, 5, 6]
    test.update_model(joint_index=joint_index)
    torque = test.action_to_torques(test_action, True)
    print(torque)
    # [ 0.0059186  -0.00672172  0.00345965 -0.00196203 -0.00203132 -0.00281999 0.00270877]