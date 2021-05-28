import numpy as np
import robosuite as suite

from .environments import ALL_ENVS
if __name__ == "__main__":

    # get the list of all environments
    envs = sorted(ALL_ENVS)

    # print info and select an environment
    print("Welcome to Surreal Robotics Suite v{}!".format(suite.__version__))
    print(suite.__logo__)
    print("Here is a list of environments in the suite:\n")

    for k, env in enumerate(envs):
        print("[{}] {}".format(k, env))
    print()
    try:
        s = input(
            "Choose an environment to run "
            + "(enter a number from 0 to {}): ".format(len(envs) - 1)
        )
        # parse input into a number within range
        k = min(max(int(s), 0), len(envs))
    except:
        print("Input is not valid. Use 0 by default.")
        k = 0

    # initialize 9the task99
    env = suite.make(
        envs[k],
        has_renderer=True,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=100,
    )
    env.reset()
    env.viewer.set_camera(camera_id=0)
    init_pos = None
    finished_pos = None

    # do visualization
    for i in range(int(5e3)):
        print(i)
        action =  np.random.randn(env.dof)
        test_action = np.array([0.1, 0., 0. , 0., 0., 0. , 0.])
        obs, reward, done, _ = env.step(test_action)
        env.render()
    
    print(f"init : {init_pos}")
    print(f"end  : {finished_pos}")
    # print(env.torque_data.shape)
    # np.savetxt('/home/wrkwak/torque_dist/withoutmassmat.csv', env.torque_data, delimiter=",")