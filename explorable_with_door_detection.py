import time
from collections import deque
import os
import cv2


os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import env.habitat.utils.pose as pu
import env.habitat.utils.visualizations as vu
import logging
from arguments import get_args
from env import make_vec_envs
from utils.storage import GlobalRolloutStorage, FIFOMemory
from utils.optimization import get_optimizer
from model import RL_Policy, Local_IL_Policy, Neural_SLAM_Module
from door_detection import Door_detection
import algo
from frontier_detection import Frontier_detection
import sys
import matplotlib
from matplotlib import pyplot as plt
from action_generation import action_generator
from topomap_construction import Topomap_construction
# from hough_door_detection import hough_detection
from env.habitat.hough_door_detection import convert_2_laser
from detr_door_detection.run_detr import run_detr
from time import time

def get_local_map_boundaries(agent_loc, local_sizes, full_sizes):
    loc_r, loc_c = agent_loc
    local_w, local_h = local_sizes
    full_w, full_h = full_sizes

    if args.global_downscaling > 1:
        gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
        gx2, gy2 = gx1 + local_w, gy1 + local_h
        if gx1 < 0:
            gx1, gx2 = 0, local_w
        if gx2 > full_w:
            gx1, gx2 = full_w - local_w, full_w

        if gy1 < 0:
            gy1, gy2 = 0, local_h
        if gy2 > full_h:
            gy1, gy2 = full_h - local_h, full_h
    else:
        gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h
    #print('new boundary {}'.format([gx1, gx2, gy1, gy2]))
    return [gx1, gx2, gy1, gy2]

explorable_threshold = 3.15
args = get_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

map_size = args.map_size_cm // args.map_resolution  # 480 pix
full_w, full_h = map_size, map_size
local_w, local_h = int(full_w / args.global_downscaling), \
                   int(full_h / args.global_downscaling)  # 240 pix

full_pose = torch.zeros(1, 3).float().cuda()
full_pose[:, :2] = args.map_size_cm / 100.0 / 2.0  # 12, 12, 0
locs = full_pose.cpu().numpy()
r, c = locs[0, 1], locs[0, 0]
loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                int(c * 100.0 / args.map_resolution)]  # 240
[gx1, gx2, gy1, gy2] = get_local_map_boundaries((loc_r, loc_c),
                                                (local_w, local_h),
                                                (full_w, full_h))
origins = [gy1 * args.map_resolution / 100.0,
           gx1 * args.map_resolution / 100.0, 0.]
start_x_gt, start_y_gt, start_o_gt = [args.map_size_cm / 100.0 / 2.0,
                                      args.map_size_cm / 100.0 / 2.0, 0.]  # 12,12,0
gt_pos = [start_x_gt - gy1 * args.map_resolution / 100.0,
          start_y_gt - gx1 * args.map_resolution / 100.0,
          start_o_gt]
x, y, o = gt_pos
x, y = x * 100.0 / 5.0, map_size//2 - y * 100.0 / 5.0
door_location = []
detected_door_list = []
raw_detect_list= []
bot_last_loc = []  # [x,y] global and pix
take_name_flag = True
scene_name = None
# cantwell
"""prior_door_list_store = [{'start': [238, 255], 'end': [228, 244]}, {'start': [252, 296], 'end': [261, 306]},
                   {'start': [266, 307], 'end': [276, 298]}, {'start': [154, 180], 'end': [143, 190]},
                   {'start': [128, 178], 'end': [138, 188]}, {'start': [107, 156], 'end': [95, 146]},
                   {'start': [187, 224], 'end': [205, 216]}, {'start': [172, 236], 'end': [160, 252]}]"""
# {'start': [187, 224], 'end': [205, 216]}, {'start': [172, 236], 'end': [160, 252]}

# dryville
"""prior_door_list = [{'start': [202, 257], 'end': [215, 263]}, {'start': [183, 272], 'end': [196, 277]},
                   {'start': [167, 243], 'end': [180, 248]}, {'start': [159, 261], 'end': [172, 266]},
                   {'start': [221, 269], 'end': [215, 282]}]"""  #

# eastville
"""prior_door_list_store = [{'start': [182, 199], 'end': [194, 209]}, {'start': [184, 324], 'end': [193, 314]},
                   {'start': [292, 266], 'end': [276, 287]}, {'start': [272, 331], 'end': [263, 343]},
                   {'start': [249, 334], 'end': [259, 343]}]"""


# prior_door_list_store.clear()  # this will lead to neartest frontier strategy


action_count = 0
cov_ratio = 0
cov_area = 0
cov_ratio_list = []
cov_area_list = []
step_list = []
log_num = 0
last_goal = None
#frontier_failed = []

established_graph = None  # if not using, equals to None
route_map = np.zeros((960, 960))
stage1_door_map = np.zeros((960, 960))  # door detection stage1
stage2_door_map = np.zeros((960, 960))  # door detection stage2
stage3_door_map = np.zeros((960, 960))  # door detection stage3


def main():
    #log_dir = "{}/models/{}/".format(args.dump_location, args.exp_name)
    #dump_dir = "{}/dump/{}/".format(args.dump_location, args.exp_name)

    #if not os.path.exists(log_dir):
    #    os.makedirs(log_dir)

    #if not os.path.exists("{}/images/".format(dump_dir)):
    #    os.makedirs("{}/images/".format(dump_dir))

    #logging.basicConfig(
    #    filename=log_dir + 'train.log',
    #    level=logging.INFO)
    #print("Dumping at {}".format(log_dir))
    print(args)
    logging.info(args)

    # Logging and loss variables
    num_scenes = args.num_processes
    num_episodes = int(args.num_episodes)
    device = args.device = torch.device("cuda:0" if args.cuda else "cpu")
    policy_loss = 0

    best_cost = 100000
    costs = deque(maxlen=1000)
    exp_costs = deque(maxlen=1000)
    pose_costs = deque(maxlen=1000)

    g_masks = torch.ones(num_scenes).float().to(device)
    l_masks = torch.zeros(num_scenes).float().to(device)

    best_local_loss = np.inf
    best_g_reward = -np.inf

    if args.eval:
        traj_lengths = args.max_episode_length // args.num_local_steps
        explored_area_log = np.zeros((num_scenes, num_episodes, traj_lengths))
        explored_ratio_log = np.zeros((num_scenes, num_episodes, traj_lengths))

    g_episode_rewards = deque(maxlen=1000)

    l_action_losses = deque(maxlen=1000)

    g_value_losses = deque(maxlen=1000)
    g_action_losses = deque(maxlen=1000)
    g_dist_entropies = deque(maxlen=1000)

    per_step_g_rewards = deque(maxlen=1000)

    g_process_rewards = np.zeros((num_scenes))

    # Starting environments
    torch.set_num_threads(1)
    envs = make_vec_envs(args)
    obs, infos = envs.reset()

    # Initialize map variables
    ### Full map consists of 4 channels containing the following:
    ### 1. Obstacle Map
    ### 2. Exploread Area
    ### 3. Current Agent Location
    ### 4. Past Agent Locations

    torch.set_grad_enabled(False)
    for scene_idx in range(1):
        scene_idx += 0

        # print('pp_list {}'.format(prior_door_list))
        # print('start_next_scene')
        frontier_detector = Frontier_detection(args.map_size_cm // args.map_resolution)
        topo = Topomap_construction()
        if established_graph:
            topo.use_exist_topomap(established_graph)
        # Calculating full and local map sizes
        map_size = args.map_size_cm // args.map_resolution
        full_w, full_h = map_size, map_size
        local_w, local_h = int(full_w / args.global_downscaling), \
                           int(full_h / args.global_downscaling)

        # Initializing full and local map
        full_map = torch.zeros(num_scenes, 4, full_w, full_h).float().to(device)
        local_map = torch.zeros(num_scenes, 4, local_w, local_h).float().to(device)

        # Initial full and local pose
        full_pose = torch.zeros(num_scenes, 3).float().to(device)
        local_pose = torch.zeros(num_scenes, 3).float().to(device)

        # Origin of local map
        origins = np.zeros((num_scenes, 3))

        # Local Map Boundaries
        lmb = np.zeros((num_scenes, 4)).astype(int)

        ### Planner pose inputs has 7 dimensions
        ### 1-3 store continuous global agent location
        ### 4-7 store local map boundaries
        planner_pose_inputs = np.zeros((num_scenes, 7))

        def init_map_and_pose():
            full_map.fill_(0.)
            full_pose.fill_(0.)
            full_pose[:, :2] = args.map_size_cm / 100.0 / 2.0

            locs = full_pose.cpu().numpy()
            planner_pose_inputs[:, :3] = locs
            for e in range(num_scenes):
                r, c = locs[e, 1], locs[e, 0]
                loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                                int(c * 100.0 / args.map_resolution)]  # position on the grid(center of local map, unit: grid)

                full_map[e, 2:, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

                lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                                  (local_w, local_h),
                                                  (full_w, full_h))

                planner_pose_inputs[e, 3:] = lmb[e]
                origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                              lmb[e][0] * args.map_resolution / 100.0,
                              0.]  # position of the bottom left (origin) of local map(unit: m)

            for e in range(num_scenes):
                local_map[e] = full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]]
                local_pose[e] = full_pose[e] - \
                                torch.from_numpy(origins[e]).to(device).float()  # position of the robot in local view

        init_map_and_pose()
        keep_exploring = True
        locs = np.array([args.map_size_cm / 100.0 / 4.0, args.map_size_cm / 100.0 / 4.0, 0])  # origins[0] [y,x,o]
        panoramic_obstacle_map = torch.zeros(num_scenes, 4, full_w, full_h).float().to(device)

        def generate_12_parts(agent_pose):
            x = agent_pose[1]  # x and y unit is m and presented in local frame
            y = agent_pose[0]
            angle = agent_pose[2]  # the angle unit is degree
            angle_list = [np.deg2rad(angle + i * 30) for i in range(12)]  # in the list, the unit becomes rad
            goal_list = []
            for content in angle_list:
                dx = 3 * np.cos(content)  # 3 * np.cos(content)
                dy = 3 * np.sin(content)  # 3 * np.sin(content)
                goal_list.append([int((x + dx) * 100 / 5), int((y + dy) * 100 / 5)])
            return goal_list

        def take_action(action, locs, first_flag=True, long_term_goal=None, room_search_flag = False):
            global action_count
            global cov_ratio
            global cov_ratio_list
            global cov_area
            global cov_area_list
            global step_list
            global detected_door_list
            global raw_detect_list
            global bot_last_loc
            global take_name_flag
            global scene_name
            global log_num
            #global frontier_failed
            global route_map
            global last_goal
            global stage1_door_map
            global stage2_door_map
            global stage3_door_map

            """if action != 4:
                action_count += 1
            else:
                action_count += 12"""
            if action != 4:
                kernel = np.ones((3, 3), np.uint8)
                obs, rew, done, infos = envs.step(torch.tensor([action]))  #
                if take_name_flag:
                    scene_name = infos[0]['scene_name']
                    scene_name = scene_name.split('/')
                    scene_name = scene_name[-1]
                    scene_name = scene_name[:-4]

                    take_name_flag = False
                if done:  # means this round should be over  action_count == args.max_episode_length - 1
                    save_path = os.path.join('/home/airs/Downloads/ANS/Neural-SLAM/613my_method_test_result', scene_name,
                                             '{}_{}.npy'.format(scene_name, scene_idx))  # scene_idx
                    save_path_time = os.path.join('/home/airs/Downloads/ANS/Neural-SLAM/613my_method_test_result', scene_name,
                                             '{}_{}_time.npy'.format(scene_name, scene_idx))  # scene_idx
                    save_path_area = os.path.join('/home/airs/Downloads/ANS/Neural-SLAM/613my_method_test_result',
                                                  scene_name,
                                                  '{}_{}_area.npy'.format(scene_name, scene_idx))  # scene_idx
                    file_path = os.path.join('/home/airs/Downloads/ANS/Neural-SLAM/613my_method_test_result',
                                             scene_name)
                    #try:
                    #    os.makedirs(file_path)
                    #except:
                    #    pass
                    print('scene name {}'.format(save_path))
                    take_name_flag = True
                    #plt.clf()
                    #plt.plot(cov_ratio_list)
                    #plt.show()
                    #np.save(save_path_time, step_list)
                    #np.save(save_path, cov_ratio_list)
                    #np.save(save_path_area, cov_area_list)
                    #np.save('zancun/step_list_cantwell_test_f.npy', step_list)
                    #np.save('zancun/cov_list_cantwell_test_f.npy', cov_ratio_list)
                    plt.ion()
                    plt.clf()
                    plt.plot(cov_ratio_list)
                    plt.show()
                    plt.pause(2)
                    plt.ioff()
                    plt.close()
                    #stage1_door_map = np.zeros_like(stage1_door_map)
                    #stage2_door_map = np.zeros_like(stage1_door_map)
                    #stage3_door_map = np.zeros_like(stage1_door_map)
                    action_count = 0
                    cov_ratio = 0
                    cov_area = 0
                    log_num = 0
                    last_goal = None
                    cov_ratio_list.clear()
                    cov_area_list.clear()
                    step_list.clear()
                    detected_door_list.clear()
                    raw_detect_list.clear()
                    bot_last_loc.clear()
                    return np.array([None]), np.array([None]), np.array([None]), False
                action_count += 1
                # print(infos[0]['sensor_pose'])
                # print(locs)
                # locs = locs + infos[0]['sensor_pose']
                locs = pu.get_new_pose(locs, infos[0]['sensor_pose'])
                absolute_locs = locs + origins[0]
                long_term_goal.reverse()  # [y, x]
                global_goal = np.array(long_term_goal) + origins[0][:2] * 100 / 5  # [y,x]
                r, c = absolute_locs[1], absolute_locs[0]
                loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                                int(c * 100.0 / args.map_resolution)]
                global_loc_xy_pix = [round(r * 100.0 / args.map_resolution),
                                     round(c * 100.0 / args.map_resolution)]
                route_map[global_loc_xy_pix[1], global_loc_xy_pix[0]] = action_count

                lmb[0] = get_local_map_boundaries((loc_r, loc_c),
                                                  (local_w, local_h),
                                                  (full_w, full_h))
                planner_pose_inputs[0, 3:] = lmb[0]
                planner_pose_inputs[0, :3] = absolute_locs
                [gx1, gx2, gy1, gy2] = lmb[0]
                origins[0] = [lmb[0][2] * args.map_resolution / 100.0,
                              lmb[0][0] * args.map_resolution / 100.0, 0.]
                goal = global_goal - origins[0][:2] * 100 / 5
                #goal = goal.astype(int).tolist()
                goal = np.round(goal).astype(int).tolist()
                goal.reverse()  # [x, y]
                locs = absolute_locs - origins[0]

                bot_last_loc.append(np.array([loc_r, loc_c]))

                # print('origin {}'.format(origins[0]))
                # print('absolute_locs {}'.format(absolute_locs))
                # print('locs {}'.format(locs))
                # goal_list = generate_12_parts(locs)
                # print(goal_list)
                angle = np.deg2rad(locs[2])
                dx = np.sin(angle)
                dy = np.cos(angle)

                #obs_show = obs[0].cpu().detach().numpy().astype(int)  # this two line is the obs from observation
                #obs_show = obs_show.transpose(1, 2, 0)

                obs_show = infos[0]['door_detection']  # this obs is from detr network after door detection

                # obs_show = run_detr(obs_show / 255)  # after door detection
                exp_ratio = infos[0]['exp_ratio']
                exp_area = infos[0]['exp_reward']*50.0  # convert to m2
                time_cost_current = time() - t_start
                #step_list.append(action_count)
                step_list.append(time_cost_current)
                cov_ratio_list.append(exp_ratio)
                cov_area_list.append(exp_area)
                """if exp_ratio:
                    cov_ratio = exp_ratio"""
                # print('exp ratio {}'.format(cov_ratio))

                # obs_show = obs_show.transpose(2,1,0)
                gt_map = infos[0]['gt_map']  # ['gt_map']
                gt_exp = infos[0]['gt_exp']
                gt_map_door = gt_map.copy()
                gt_exp_door = gt_exp.copy()
                # gt_map = gt_map[::-1, ::-1]
                # gt_map = gt_map[::-1,:]  # flip vertically
                # gt_map = gt_map[:, ::-1]  # flip horizontally
                gt_map_local_grid = np.rint(gt_map[gx1:gx2, gy1:gy2])
                gt_exp_local_grid = np.rint(gt_exp[gx1:gx2, gy1:gy2])
                gt_map = gt_map.transpose()
                gt_exp = gt_exp.transpose()
                # ---------------------------------------------
                """planner_pose_inputs_copy = planner_pose_inputs.copy()
                planner_inputs = [{} for e in range(num_scenes)]
                for e, p_input in enumerate(planner_inputs):
                    p_input['goal'] = goal
                    p_input['map_pred'] = gt_map_local_grid
                    p_input['exp_pred'] = gt_exp_local_grid
                    p_input['pose_pred'] = planner_pose_inputs[0]
                    p_input['mid_out'] = True
                path_list = []
                for _ in range(150):
                    output = envs.get_short_term_goal(planner_inputs)
                    # print(output)
                    dist = output[0][0].cpu().detach().numpy()
                    stg = output[0][1:].cpu().detach().numpy()
                    stg_x, stg_y = stg
                    # global_stg = np.array([stg_y, stg_x]) + np.array([origins[0][0] * 100 / 5, origins[0][1] * 100 / 5])
                    # global_stg = np.array([stg_y, stg_x]) + np.array([origins[0][0] * 100 / 5, origins[0][1] * 100 / 5])
                    path_list.append([int(stg_x + origins[0][1] * 100 / 5), int(stg_y + origins[0][0] * 100 / 5)])
                    # print(total_dist)
                    dist2goal = pu.get_l2_distance(stg_x, goal[0], stg_y, goal[1])
                    # if dist2goal < 1:
                    #    break

                    planner_pose_inputs[0, :2] = stg_y * 5 / 100 + origins[0][0], stg_x * 5 / 100 + origins[0][1]
                    for e, p_input in enumerate(planner_inputs):
                        p_input['goal'] = goal
                        p_input['map_pred'] = gt_map_local_grid
                        p_input['exp_pred'] = gt_exp_local_grid
                        p_input['pose_pred'] = planner_pose_inputs[0]
                        p_input['mid_out'] = True
                planner_pose_inputs[0, :2] = planner_pose_inputs_copy[0, :2]"""
                # ---------------------------------------------------

                planner_inputs = [{} for e in range(num_scenes)]

                if room_search_flag:

                    for door_detected in detected_door_list:
                        door_start = door_detected['start']
                        door_end = door_detected['end']
                        gt_map_door = cv2.line(gt_map_door, (door_start[1], door_start[0]), (door_end[1], door_end[0]), 1, thickness = 1)
                        gt_exp_door = cv2.line(gt_exp_door, (door_start[1], door_start[0]), (door_end[1], door_end[0]),
                                               0, thickness=1)
                    gt_map_local_grid = np.rint(gt_map_door[gx1:gx2, gy1:gy2])
                    gt_exp_local_grid = np.rint(gt_exp_door[gx1:gx2, gy1:gy2])
                    #plt.imshow(gt_map_local_grid)
                    #plt.show()
                for e, p_input in enumerate(planner_inputs):
                    p_input['goal'] = goal
                    p_input['map_pred'] = gt_map_local_grid
                    p_input['exp_pred'] = gt_exp_local_grid
                    p_input['pose_pred'] = planner_pose_inputs[0]
                    p_input['mid_out'] = True
                output = envs.get_short_term_goal(planner_inputs)
                stg = output[0][1:].cpu().detach().numpy()

                # stg = stg.tolist()  # under local frame
                stg_x, stg_y = stg
                """plt.imshow(gt_map_local_grid.transpose())
                plt.plot(locs[1] * 100 / 5, locs[0] * 100 / 5, 'o', color='pink')
                plt.plot(goal[0], goal[1], 'o', color='red')

                for i in path_list:
                    plt.plot(i[0]-origins[0][1]*100/5, i[1]-origins[0][0]*100/5, 'o', color = 'plum')
                plt.plot(stg_x, stg_y, 'o', color='blue')
                plt.arrow(locs[1] * 100 / 5, locs[0] * 100 / 5, dx * 8, dy * (8 * 1.25), head_width=8,
                          head_length=8 * 1.25,
                          length_includes_head=True, fc='Red', ec='Red', alpha=0.9)
                plt.plot(locs[1] * 100 / 5, locs[0] * 100 / 5, 'o', color = 'white')
                plt.show()
                plt.clf()"""
                # stg_x = stg_x + origins[0][1]*100/args.map_resolution
                # stg_y = stg_y + origins[0][0]*100/args.map_resolution
                stg = [stg_x * args.map_resolution / 100, stg_y * args.map_resolution / 100]
                plt.ion()

                plt.clf()
                plt.cla()
                plt.subplot(1, 2, 1)
                plt.imshow(gt_map + gt_exp)
                plt.arrow(absolute_locs[1] * 100 / 5, absolute_locs[0] * 100 / 5, dx * 8, dy * (8 * 1.25), head_width=8,
                          head_length=8 * 1.25,
                          length_includes_head=True, fc='Red', ec='Red',
                          alpha=0.9)  # absolute_locs[1]*100/5, absolute_locs[0]*100/5
                # plt.plot(origins[0][0]*100/5,origins[0][1]*100/5,'o',color='red')
                plt.plot(lmb[0][0], lmb[0][2], 'o', color='red')
                plt.plot(lmb[0][1], lmb[0][2], 'o', color='red')
                plt.plot(lmb[0][0], lmb[0][3], 'o', color='red')
                plt.plot(lmb[0][1], lmb[0][3], 'o', color='red')

                plt.plot(global_goal[1], global_goal[0], 'o', color='lime')

                for door in detected_door_list:
                    plt.plot([door['start'][0], door['end'][0]], [door['start'][1], door['end'][1]], color='fuchsia')
                    # plt.plot(door[0], door[1], 'o', color='pink')
                for door in raw_detect_list:
                    plt.plot(door[0], door[1], 'o', color='pink')

                # for goal in goal_list:
                #    plt.plot(goal[0]+origins[0][1]*100/5,goal[1]+origins[0][0]*100/5,'o',color='yellow')
                """for i in path_list:
                    plt.plot(i[0], i[1], 'o', color = 'plum')"""
                #plt.plot((stg[0] + origins[0][1]) * 100 / 5, (stg[1] + origins[0][0]) * 100 / 5, 'o', color='aqua')

                #plt.savefig('hough_result.png')
                plt.subplot(1, 2, 2)
                plt.imshow(obs_show)  # , cmap='gray'
                # plt.show()
                plt.pause(0.1)
                plt.ioff()

                return locs, stg, goal, False
            else:
                action = 1  # 12 turn_right makes up on scan motion
                for turn_num in range(12):
                    obs, rew, done, infos = envs.step(torch.tensor([action]))  #
                    if take_name_flag:
                        scene_name_full = infos[0]['scene_name']
                        scene_name = scene_name_full.split('/')
                        scene_name = scene_name[-1]
                        scene_name = scene_name[:-4]
                        take_name_flag = False
                    action_count += 1
                    exp_ratio = infos[0]['exp_ratio']
                    exp_area = infos[0]['exp_reward'] * 50.0  # convert to m2
                    time_cost_current = time() - t_start
                    # step_list.append(action_count)
                    step_list.append(time_cost_current)
                    cov_ratio_list.append(exp_ratio)
                    cov_area_list.append(exp_area)
                    if done:  # means this round should be over  action_count == args.max_episode_length - 1

                        save_path = os.path.join('/home/airs/Downloads/ANS/Neural-SLAM/613my_method_test_result',
                                                 scene_name,
                                                 '{}_{}.npy'.format(scene_name, scene_idx))  # scene_idx
                        save_path_time = os.path.join('/home/airs/Downloads/ANS/Neural-SLAM/613my_method_test_result',
                                                 scene_name,
                                                 '{}_{}_time.npy'.format(scene_name, scene_idx))  # scene_idx
                        save_path_area = os.path.join('/home/airs/Downloads/ANS/Neural-SLAM/613my_method_test_result',
                                                 scene_name,
                                                 '{}_{}_area.npy'.format(scene_name, scene_idx))  # scene_idx
                        file_path = os.path.join('/home/airs/Downloads/ANS/Neural-SLAM/613my_method_test_result',
                                                 scene_name)
                        #try:
                        #    os.makedirs(file_path)
                        #except:
                        #    pass
                        print('scene name {}'.format(save_path))
                        take_name_flag = True
                        #plt.clf()
                        #plt.plot(cov_ratio_list)
                        #plt.show()
                        #np.save(save_path, cov_ratio_list)
                        #np.save(save_path_area, cov_area_list)
                        #np.save(save_path_time, step_list)
                        #np.save('zancun/step_list_cantwell_test_f.npy', step_list)
                        #np.save('zancun/cov_list_cantwell_test_f.npy', cov_ratio_list)
                        plt.ion()
                        plt.clf()
                        plt.plot(cov_ratio_list)
                        plt.show()
                        plt.pause(2)
                        plt.ioff()
                        plt.close()
                        #stage1_door_map = np.zeros_like(stage1_door_map)
                        #stage2_door_map = np.zeros_like(stage1_door_map)
                        #stage3_door_map = np.zeros_like(stage1_door_map)
                        action_count = 0
                        cov_ratio = 0
                        cov_area = 0
                        log_num = 0
                        last_goal = None
                        cov_ratio_list.clear()
                        cov_area_list.clear()
                        step_list.clear()
                        detected_door_list.clear()
                        raw_detect_list.clear()
                        bot_last_loc.clear()
                        return np.array([None]), np.array([None]), np.array([None]), False
                    # print(infos[0]['sensor_pose'])
                    # print(locs)
                    locs = pu.get_new_pose(locs, infos[0]['sensor_pose'])


                    absolute_locs = locs + origins[0]
                    r, c = absolute_locs[1], absolute_locs[0]
                    loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                                    int(c * 100.0 / args.map_resolution)]

                    lmb[0] = get_local_map_boundaries((loc_r, loc_c),
                                                      (local_w, local_h),
                                                      (full_w, full_h))
                    [gx1, gx2, gy1, gy2] = lmb[0]
                    planner_pose_inputs[0, 3:] = lmb[0]
                    planner_pose_inputs[0, :3] = absolute_locs
                    origins[0] = [lmb[0][2] * args.map_resolution / 100.0,
                                  lmb[0][0] * args.map_resolution / 100.0, 0.]
                    locs = absolute_locs - origins[0]
                    # print('origin {}'.format(origins[0]))
                    # print('absolute_locs {}'.format(absolute_locs))
                    # print('locs {}'.format(locs))

                    # print(locs)
                    angle = np.deg2rad(locs[2])
                    dx = np.sin(angle)
                    dy = np.cos(angle)

                    #obs_show = obs[0].cpu().detach().numpy().astype(np.uint8)
                    #obs_show = obs_show.transpose(1, 2, 0)

                    obs_show = infos[0]['door_detection']  # this obs is from detr network after door detection
                    depth_image = infos[0]['depth']

                    #np.save('paper_fig/{}.npy'.format(turn_num), obs[0].cpu().detach().numpy().astype(np.uint8).transpose(1, 2, 0))
                    #np.save('paper_fig/depth{}.npy'.format(turn_num),
                    #        depth_image)
                    plt.ion()
                    plt.clf()
                    plt.cla()
                    plt.imshow(obs_show)  # , cmap='gray'
                    #plt.show()
                    plt.pause(0.1)
                    plt.ioff()
                plt.close()
                    # plt.imsave('pic_save/{}.png'.format(action_count), obs_show)
                # print('origin {}'.format(origins[0]))
                # print('absolute_locs {}'.format(absolute_locs))
                # print('locs {}'.format(locs))
                # gt_map = infos[0]['pano_map']  # ['gt_map']
                # gt_exp = infos[0]['pano_exp']
                #cov_ratio_list.append(infos[0]['exp_ratio'])
                #cov_area_list.append(infos[0]['exp_reward']*50.0)
                gt_map = infos[0]['gt_map']  # ['gt_map']
                gt_exp = infos[0]['gt_exp']
                gt_pano_map = infos[0]['pano_map']
                gt_pano_exp = infos[0]['pano_exp']
                gt_door_map = infos[0]['door_obs_map']
                gt_door_local_map = infos[0]['door_local_map']  # this is for door detection

                # obs_show = hough_detection(obs_show)

                if exp_ratio:
                    cov_ratio = exp_ratio
                if exp_area:
                    cov_area = exp_area
                # print('exp ratio {}'.format(cov_ratio))
                # gt_map = gt_map.transpose()
                # gt_exp = gt_exp.transpose()

                # gt_map_local_grid = np.rint(gt_map[gy1:gy2, gx1:gx2])
                # gt_exp_local_grid = np.rint(gt_exp[gy1:gy2, gx1:gx2])
                gt_map_local_grid = np.rint(gt_map[gx1:gx2, gy1:gy2])
                gt_exp_local_grid = np.rint(gt_exp[gx1:gx2, gy1:gy2])

                # plt.imshow(gt_map_local_grid.transpose())
                # plt.plot(locs[1]*100/5, locs[0]*100/5, 'o', color = 'green')
                # plt.show()
                # exit()
                gt_map = gt_map.transpose()
                gt_exp = gt_exp.transpose()
                gt_pano_map = gt_pano_map.transpose()
                gt_pano_exp = gt_pano_exp.transpose()
                gt_door_map = gt_door_map.transpose()
                gt_door_local_map = gt_door_local_map.transpose()
                """plt.subplot(1,2,1)
                plt.imshow(gt_pano_map)
                plt.subplot(1,2,2)
                plt.imshow(gt_door_local_map)
                plt.show()"""
                #np.save('/home/airs/Downloads/ANS/Neural-SLAM/topo_mapping_result/Hambleton/{}/gt_map_Hambleton.npy'.format(scene_idx), gt_map)#scene_idx
                #np.save('/home/airs/Downloads/ANS/Neural-SLAM/topo_mapping_result/Hambleton/{}/gt_exp_Hambleton.npy'.format(scene_idx), gt_exp)#scene_idx
                #np.save('/home/airs/Downloads/ANS/Neural-SLAM/map_save_for_dude/Hambleton/{}/gt_map_{}.npy'.format(scene_idx,
                #    action_count), gt_map)
                #np.save('/home/airs/Downloads/ANS/Neural-SLAM/map_save_for_dude/Hambleton/{}/gt_exp_{}.npy'.format(scene_idx,
                #    action_count), gt_exp)
                #np.save('/home/airs/Downloads/ANS/Neural-SLAM/paper_fig/gt_map.npy', gt_map)
                #np.save('/home/airs/Downloads/ANS/Neural-SLAM/paper_fig/gt_exp.npy', gt_exp)
                """np.save('/home/airs/Downloads/ANS/Neural-SLAM/map_for_time_test/Hambleton/{}/gt_map_{}.npy'.format(
                    scene_idx,
                    action_count), gt_map)
                np.save('/home/airs/Downloads/ANS/Neural-SLAM/map_for_time_test/Hambleton/{}/gt_exp_{}.npy'.format(
                    scene_idx,
                    action_count), gt_exp)"""
                #np.save('route_save/{}.npy'.format(action_count), route_map)
                #np.save('paper_fig/route_map.npy'.format(action_count), route_map)
                # print(planner_pose_inputs)
                c, r = absolute_locs[:-1]
                global_loc_xy_pix = [round(r * 100.0 / args.map_resolution),
                                     round(c * 100.0 / args.map_resolution)]
                """start = [round(r * 100.0 / args.map_resolution - gx1),
                         round(c * 100.0 / args.map_resolution - gy1)]"""
                goal_list = generate_12_parts(locs)
                planner_pose_inputs_copy = planner_pose_inputs.copy()
                unexplorable_list = []
                create_door_detection = True
                # door_detect = Door_detection(list_x, list_y)
                full_list_x = []
                full_list_y = []
                for j, goal in enumerate(goal_list):
                    # goal = [int(302-origins[0][1]*100/5), int(237-origins[0][0]*100/5)]  # [x, y]
                    # goal = [182, 117]
                    # print(goal)

                    planner_inputs = [{} for e in range(num_scenes)]
                    for e, p_input in enumerate(planner_inputs):
                        p_input['goal'] = goal
                        p_input['map_pred'] = gt_map_local_grid
                        p_input['exp_pred'] = gt_exp_local_grid
                        p_input['pose_pred'] = planner_pose_inputs[0]
                        p_input['mid_out'] = True
                    loc_y = (planner_pose_inputs[0][0] - origins[0][0]) * 100 / 5
                    loc_x = (planner_pose_inputs[0][1] - origins[0][1]) * 100 / 5
                    total_dist = 0

                    list_x = []
                    list_y = []
                    for _ in range(150):  # 150
                        output = envs.get_short_term_goal(planner_inputs)
                        # print(output)
                        dist = output[0][0].cpu().detach().numpy()
                        total_dist += dist
                        stg = output[0][1:].cpu().detach().numpy()
                        stg_x, stg_y = stg
                        # global_stg = np.array([stg_y, stg_x]) + np.array([origins[0][0] * 100 / 5, origins[0][1] * 100 / 5])
                        list_x.append(stg_x + origins[0][1] * 100 / 5)
                        list_y.append(stg_y + origins[0][0] * 100 / 5)
                        full_list_x.append(stg_x + origins[0][1] * 100 / 5)
                        full_list_y.append(stg_y + origins[0][0] * 100 / 5)
                        # print(total_dist)
                        dist2goal = pu.get_l2_distance(stg_x, goal[0], stg_y, goal[1])
                        if total_dist > explorable_threshold:
                            if dist2goal > 3:  # remove the point that the distance between goal and final pos larger than 3 pix
                                unexplorable_list.append(goal)
                                break
                        if dist2goal < 1:
                            break

                        planner_pose_inputs[0, :2] = stg_y * 5 / 100 + origins[0][0], stg_x * 5 / 100 + origins[0][1]
                        for e, p_input in enumerate(planner_inputs):
                            p_input['goal'] = goal
                            p_input['map_pred'] = gt_map_local_grid
                            p_input['exp_pred'] = gt_exp_local_grid
                            p_input['pose_pred'] = planner_pose_inputs[0]
                            p_input['mid_out'] = True
                    # print(total_dist)
                    if total_dist == 0.0:
                        # print('goal in obstacle')
                        unexplorable_list.append(goal)
                    planner_pose_inputs[0, :2] = planner_pose_inputs_copy[0, :2]

                    if create_door_detection and first_flag:
                        global door_detect
                        door_detect = Door_detection(list_x, list_y)
                        create_door_detection = False
                    else:
                        door_detect.new_list(list_x, list_y)
                    if create_door_detection and not first_flag:
                        door_detect.reset(list_x, list_y)
                        create_door_detection = False
                    # np.save('list_y{}.npy'.format(j), list_y)
                    # np.save('list_x{}.npy'.format(j), list_x)

                # global_stg = np.array([stg_y, stg_x]) + np.array([origins[0][0]*100/5, origins[0][1]*100/5])
                # print(global_stg)
                # print(dist)
                # start = pu.threshold_poses(start, gt_map_local_grid.shape)
                # stg = envs._get_stg(gt_map_local_grid, gt_exp_local_grid, start, goal, [gx1, gx2, gy1, gy2])

                # gt_map = gt_map[::-1, :]  # filp vertically
                # gt_map = gt_map[::-1, ::-1]
                # exit()
                # raw_door_list = door_detect.get_door_point()

                # following two lines is using 12 points method to detect door
                # door_list, raw_list = door_detect.door_filter(gt_map)
                # detected_door_list.extend(door_list)
                # print(door_list)

                hough_door_list, laser_list = convert_2_laser(gt_pano_map, gt_pano_exp, absolute_locs)

                #plt.clf()
                #plt.subplot(1,2,1)
                #plt.ion()
                #plt.subplot(1,2,1)
                #plt.imshow(gt_door_map)
                for grid_idx in hough_door_list:
                    #plt.plot(grid_idx[0], grid_idx[1], 'o', color='red')
                    stage1_door_map[grid_idx[1], grid_idx[0]] = 1
                """plt.subplot(1,2,2)
                plt.imshow(gt_map)
                for grid_idx in hough_door_list:
                    plt.plot(grid_idx[0], grid_idx[1], 'o', color='red')"""
                #plt.subplot(1,2,2)
                #plt.imshow(stage1_door_map)
                #plt.show()
                #plt.pause(1)
                #plt.ioff()
                #plt.close()

                # following method is using RGB to filter the door
                filterd_hough_list = []
                for grid_idx in hough_door_list:

                    if np.sum(gt_door_map[grid_idx[1] - 2:grid_idx[1] + 3, grid_idx[0] - 2:grid_idx[0] + 3]) > 0:
                        filterd_hough_list.append(grid_idx)
                        stage2_door_map[grid_idx[1], grid_idx[0]] = 1

                door_list, raw_list = \
                    door_detect.door_filter(gt_door_local_map, gt_map, gt_exp, global_loc_xy_pix, bot_last_loc,
                                            detected_door_list,
                                            use_12point=False,
                                            external_door_point=filterd_hough_list)

                # uncomment these two is frontier method
                #door_list.clear()
                #raw_list.clear()


                bot_last_loc.clear()
                close_door_list = []
                close_door_list.extend(door_list)
                detected_door_list.extend(door_list)
                raw_detect_list.extend(raw_list)
                #np.save('door_dict_main.npy', detected_door_list)


                """for door in prior_door_list:
                    mid_point = door['mid']
                    start = door['start']
                    end = door['end']
                    print('---')
                    print('current_door {}'.format(door))
                    print('dist {}'.format(pu.get_l2_distance(r * 100 / 5, mid_point[0], c * 100 / 5, mid_point[1])))
                    print('rob loc {}, {}'.format(r * 100 / 5, c * 100 / 5))
                    print('mid {}'.format(mid_point))
                    print('gt_exp {}'.format(gt_exp[int(mid_point[1]), int(mid_point[0])]))
                    print('gt_map_start {}'.format(np.sum(gt_map[start[1]-1:start[1]+2, start[0]-1:start[0]+2])))
                    print('gt_map_end {}'.format(np.sum(gt_map[end[1]-1:end[1]+2, end[0]-1:end[0]+2])))
                    print('---')
                    if pu.get_l2_distance(r * 100 / 5, mid_point[0], c * 100 / 5, mid_point[1]) <= args.vision_range \
                            and gt_exp[int(mid_point[1]), int(mid_point[0])] == 1 \
                            and np.sum(gt_map[start[1] - 2:start[1] + 3, start[0] - 2:start[0] + 3]) > 1 \
                            and np.sum(
                        gt_map[end[1] - 2:end[1] + 3, end[0] - 2:end[0] + 3]) > 1:  # if cantwell this line > 1

                        # print('find door {}'.format(door))
                        close_door_list.append(door)"""

                        # pass
                #for door in close_door_list:  # remove the chosen door
                #    prior_door_list.remove(door)
                # close_door_list = [{'start': [238, 255], 'end': [228, 244], 'mid': [233., 249.5]}]
                # print(planner_pose_inputs[0, :2]*100/5)
                #if established_graph:
                #    close_door_list = prior_door_list_store

                current_loc = planner_pose_inputs[0,
                              :2] * 100 / args.map_resolution  # add this line, means searching from robot's current location
                #f_start_time = time()
                f_list, info_gain_list, show_map, room_exp_list, door_grid = frontier_detector.frontier_detection(np.array(current_loc),
                                                                                       planner_pose_inputs[0,
                                                                                       :2] * 100 / args.map_resolution,
                                                                                       np.rint(gt_map), np.rint(gt_exp),
                                                                                       lmb[0],
                                                                                       detected_door_list,
                                                                                       laser_list
                                                                                       )  # [y, x] close_door_list
                #step_list.append(time()-f_start_time)
                detected_door_list = topo.same_node_check(room_exp_list, detected_door_list)
                # combine the node for loop case, common this if pure frontier
                door_list, door_remove_list = topo.check_topomap(door_list, detected_door_list,
                                                                 room_exp_list, current_loc, gt_map, gt_exp, lmb[0], door_grid, scene_idx, scene_name, laser_list)

                # these two for pure frontier
                #door_list = []
                #door_remove_list = []
                # check topomap checks the detected door's relation with the current node
                in_point, return_flag = topo.add_room(door_list, [absolute_locs[1] * 100 / 5, absolute_locs[0] * 100 / 5],
                                         gt_map, gt_exp, lmb[0])
                in_point.reverse()  # in point is the entry of the current node, this can make sure that the searching range
                # is within the current room even if the robot is going out of the current room

                for door in detected_door_list:
                    stage3_door_map[door['start'][1], door['start'][0]] = 1
                    stage3_door_map[door['end'][1], door['end'][0]] = 1

                for door in door_remove_list:  # common this for if pure frontier

                    detected_door_list.remove(door)  # remove the door that does not enclose
                    stage3_door_map[door['end'][1], door['end'][0]] = 0
                    stage3_door_map[door['start'][1], door['start'][0]] = 0
                    #door_detect.door_remove(door)  # remove the door in door detect class

                """plt.ion()
                plt.subplot(1, 3, 1)
                plt.imshow(stage1_door_map)
                plt.subplot(1, 3, 2)
                plt.imshow(stage2_door_map)
                plt.subplot(1, 3, 3)
                plt.imshow(stage3_door_map)
                plt.show()
                plt.pause(5)
                plt.ioff()
                plt.close()"""
                print('num in stg3 {}'.format(np.sum(stage3_door_map)))

                print("return_flag {}".format(return_flag))
                if not return_flag:
                    log_num += 1
                    plt.clf()
                    #fig_save_path = '/home/airs/Downloads/ANS/Neural-SLAM/F_method_log/{}/{}'.format(scene_name,scene_idx)
                    fig_save_path = '/home/airs/Downloads/ANS/Neural-SLAM/613f_method_log/{}/{}'.format(scene_name,
                                                                                                     scene_idx)
                    try:
                        #pass
                        os.makedirs(fig_save_path)
                    except:
                        pass
                    plt.imshow(show_map)
                    plt.plot(absolute_locs[1]*100/5, absolute_locs[0]*100/5, 'o', color = 'red')
                    for f in f_list:
                        plt.plot(f[1], f[0], 'o', color = 'peru')
                    #plt.savefig(fig_save_path+'/{}.png'.format(log_num))



                    final_goal = None
                    shortest_dist = 10000000

                    if len(f_list) == 1:  # choose the nearest frontier to go
                        final_goal_ = f_list[0]
                        """bot_goal_dist = pu.get_l2_distance(final_goal_[1],
                                                           planner_pose_inputs[0, 1] * 100 / args.map_resolution,
                                                           final_goal_[0],
                                                           planner_pose_inputs[0, 0] * 100 / args.map_resolution)"""
                        final_goal_ = final_goal_ - origins[0, :2] * 100 / args.map_resolution
                        final_goal_ = final_goal_.astype(int).tolist()  # convert into local frame

                        if final_goal_ == last_goal:
                            final_goal = None

                        else:
                            final_goal = final_goal_
                        #if bot_goal_dist <= 30:
                        #    final_goal = None
                    elif len(f_list) == 0:
                        pass
                    else:
                        for f_idx, goal_frontier in enumerate(f_list):
                            goal_frontier = goal_frontier.astype(int).tolist()
                            info_gain = info_gain_list[f_idx]
                            bot_goal_dist = pu.get_l2_distance(goal_frontier[1],
                                                               planner_pose_inputs[0, 1] * 100 / args.map_resolution,
                                                               goal_frontier[0],
                                                               planner_pose_inputs[0, 0] * 100 / args.map_resolution)
                            #print('f_dist {}'.format(bot_goal_dist))
                            #print('info_gain {}'.format(info_gain))
                            #bot_goal_dist = bot_goal_dist/info_gain

                            w1 = 1
                            w2 = 2
                            bot_goal_dist = w1*bot_goal_dist - w2*info_gain
                            #print('cost value {}'.format(bot_goal_dist))
                            if bot_goal_dist < shortest_dist:# and bot_goal_dist > 30:  # 30 is the bot near range in frontier detection

                                final_goal_ = goal_frontier
                                final_goal_ = final_goal_ - origins[0, :2] * 100 / args.map_resolution
                                final_goal_ = final_goal_.astype(int).tolist()

                                if final_goal_ == last_goal:

                                    final_goal = None
                                else:
                                    shortest_dist = bot_goal_dist
                                    final_goal = final_goal_
                else:

                    final_goal = in_point - origins[0, :2] * 100 / args.map_resolution
                    final_goal = final_goal.astype(int).tolist()
                try:
                    last_goal = final_goal.copy()

                except:
                    last_goal = None
                #print('cov_ ration {}'.format(cov_ratio))
                #if cov_ratio > 0.99:
                #    final_goal = None  # only use this during pure frontier method, prevent stuck

                if final_goal:
                    # print('the chosen frontier is {}[y,x]'.format(final_goal))
                    final_goal.reverse()  # [x, y]
                    # print('final_goal{}'.format(final_goal))
                    planner_inputs_frontier = [{} for e in range(num_scenes)]
                    for e, p_input in enumerate(planner_inputs_frontier):
                        p_input['goal'] = final_goal
                        p_input['map_pred'] = gt_map_local_grid
                        p_input['exp_pred'] = gt_exp_local_grid
                        p_input['pose_pred'] = planner_pose_inputs[0]
                        p_input['mid_out'] = True
                    output = envs.get_short_term_goal(planner_inputs_frontier)
                    stg = output[0][1:].cpu().detach().numpy()
                    # stg = stg.tolist()  # under local frame
                    stg_x, stg_y = stg
                    # stg_x = stg_x + origins[0][1]*100/args.map_resolution
                    # stg_y = stg_y + origins[0][0]*100/args.map_resolution
                    stg = [stg_x * args.map_resolution / 100,
                           stg_y * args.map_resolution / 100]  # [x, y]  may be should change xy
                    # print('short term goal is {}'.format(stg))
                    # del door_detect

                    # print(len(goal_list)-len(unexplorable_list))
                    # print(unexplorable_list)
                    """plt.clf()
                    plt.subplot(1, 2, 1)
                    plt.imshow(gt_map + gt_exp)
                    plt.plot(lmb[0][0], lmb[0][2], 'o', color='green')
                    plt.plot(lmb[0][1], lmb[0][2], 'o', color='green')
                    plt.plot(lmb[0][0], lmb[0][3], 'o', color='green')
                    plt.plot(lmb[0][1], lmb[0][3], 'o', color='green')
                    plt.arrow(absolute_locs[1] * 100 / 5, absolute_locs[0] * 100 / 5, dx * 8, dy * (8 * 1.25), head_width=8,
                              head_length=8 * 1.25,
                              length_includes_head=True, fc='Red', ec='Red',
                              alpha=0.9)  # absolute_locs[1]*100/5, absolute_locs[0]*100/5
                    plt.plot(final_goal[0] + origins[0][1] * 100 / 5, final_goal[1] + origins[0][0] * 100 / 5, 'o',
                             color='lime')
                    plt.plot(full_list_x, full_list_y, 'o', color='blue')
                    plt.plot((stg[0] + origins[0][1]) * 100 / 5, (stg[1] + origins[0][0]) * 100 / 5, 'o', color='aqua')
                    # np.save('list_x.npy', list_x)
                    # np.save('list_y.npy', list_y)

                    for door in door_list:
                        plt.plot([door['start'][0], door['end'][0]], [door['start'][1], door['end'][1]], color='fuchsia')
                        # plt.plot(door['end'][0], door['end'][1], 'o', color='pink')
                    for door in raw_list:
                        plt.plot(door[0], door[1], 'o', color='pink')

                    for goal in goal_list:
                        if goal not in unexplorable_list:
                            plt.plot(goal[0] + origins[0][1] * 100 / 5, goal[1] + origins[0][0] * 100 / 5, 'o', color='red')
                        else:
                            plt.plot(goal[0] + origins[0][1] * 100 / 5, goal[1] + origins[0][0] * 100 / 5, 'o',
                                     color='cyan')
                    plt.subplot(1, 2, 2)
                    plt.imshow(obs_show, cmap='gray')
                    plt.show()"""
                else:
                    stg = None
            return locs, stg, final_goal, return_flag

        def go2goal(locs, stg, long_term_goal, first_flag, achieve_criterion=10, consider_door = False):
            # locs[y, x o]/ stg input [x, y] local, m; long term goal local pix [x,y]
            # achieve_criterion = 10
            # consider_door if true, consider door when path planning
            if stg:
                stg.reverse()
                action_value = action_generator(locs, stg)
            else:
                action_value = 0  # defualt action: turn_left
            achieve_flag = False
            if pu.get_l2_distance(long_term_goal[0], (locs[1]) * 100 / 5, long_term_goal[1], (locs[
                0]) * 100 / 5) > achieve_criterion:  # pu.get_l2_distance(long_term_goal[0], 120, long_term_goal[1], 120)
                #print('dist to goal {}'.format(pu.get_l2_distance(long_term_goal[0], (locs[1])*100/5, long_term_goal[1], (locs[0])*100/5)))
                # means not achieving the goal
                locs, stg, long_term_goal, _ = take_action(action_value, locs, first_flag, long_term_goal, consider_door)
            else:
                achieve_flag = True
            return locs, stg, long_term_goal, achieve_flag

        def generate_return_list(visitied_list, exit_point):  # both [x, y], global, pix
            visitied_list.reverse()  # calculate from current position
            shortest_dist = 10000
            for idx, visitied_point in enumerate(visitied_list):
                dist_to_door = pu.get_l2_distance(visitied_point[0], exit_point[0], visitied_point[1], exit_point[1])
                if dist_to_door < shortest_dist:
                    shortest_dist = dist_to_door
                    shortest_idx = idx + 1
            crop_list = visitied_list[:shortest_idx]
            filter_list = []
            for i in range(len(crop_list) - 1):
                if pu.get_l2_distance(crop_list[i][0], crop_list[i + 1][0], crop_list[i][1],
                                      crop_list[i + 1][1]) > 0.18 * 100 / 5:
                    filter_list.append(crop_list[i])
            filter_list.append(exit_point)
            return filter_list

        def room_searching(locs, stg, long_term_goal, first_flag, whether_returning):
            if long_term_goal:
                achieve_flag = False
                room_search_flag = True
                step_limit = 20  # 20 for my method
                step_count = 0

                while room_search_flag:  # this is the room searching part
                    while not achieve_flag:
                        if not whether_returning:
                            locs, stg, long_term_goal, achieve_flag = go2goal(locs, stg, long_term_goal, first_flag, consider_door=True)
                        else:
                            locs, stg, long_term_goal, achieve_flag = go2goal(locs, stg, long_term_goal, first_flag,
                                                                              consider_door=False)
                        if not locs.any():
                            return locs, stg, long_term_goal
                        step_count += 1
                        # dist = pu.get_l2_distance(120, long_term_goal[0], 120, long_term_goal[1])
                        if step_count > step_limit:
                            print('failed to achieve')
                            step_count = 0
                            break
                        if achieve_flag:
                            print('achieved')
                            step_count = 0

                    achieve_flag = False
                    locs, stg, long_term_goal, whether_returning = take_action(4, locs, first_flag)
                    if not locs.any():
                        return locs, stg, long_term_goal
                    print('long tt {}'.format(long_term_goal))
                    if not long_term_goal:
                        room_search_flag = False
                        print('room search done')
            return locs, stg, long_term_goal

        def room_moving(locs, stg, long_term_goal, first_flag):
            achieve_flag = False
            exit_goal_list = topo.choose_door([(locs[0] + origins[0][0]) * 100 / args.map_resolution,
                                               (locs[1] + origins[0][1]) * 100 / args.map_resolution])
            # return_list = generate_return_list(visited_waypoint, exit_goal)
            print('exit_goal {}'.format(exit_goal_list))
            return_step = 0
            return_threshold = 100#60
            for exit_goal in exit_goal_list:
                # for return_waypoint in return_list[1:]:
                long_term_goal = [exit_goal[0] - origins[0][1] * 100 / args.map_resolution,
                                  exit_goal[1] - origins[0][0] * 100 / args.map_resolution]  # convert to local frame
                # print(long_term_goal)
                stg = [long_term_goal[0] * 5 / 100, long_term_goal[1] * 5 / 100]
                while not achieve_flag:
                    locs, stg, long_term_goal, achieve_flag = go2goal(locs, stg, long_term_goal, first_flag,
                                                                      achieve_criterion=5)  # default is 5
                    if not locs.any():
                        return locs, stg, long_term_goal
                    return_step += 1
                    # dist = pu.get_l2_distance(120, long_term_goal[0], 120, long_term_goal[1])
                    if return_step > return_threshold:
                        print('failed to achieve')
                        return_step = 0
                        break
                    if achieve_flag:
                        print('achieved')
                        return_step = 0
                achieve_flag = False
            return locs, stg, long_term_goal

        def exploration(locs):
            global t_start
            t_start = time()

            print('locs in exp {}'.format(locs))
            # auto exploration stage

            first_flag = True  # to check whether it is the first scan
            # take the initialization step, first we scan the surrounding
            locs, stg, long_term_goal, whether_returning = take_action(4, locs, first_flag)  # long term goal is under local frame

            if not locs.any():
                return None
            print('first long term goal {}'.format(long_term_goal))
            first_flag = False
            print(topo.stop_exp())
            while not topo.stop_exp():

                # first searching the current room
                locs, stg, long_term_goal = room_searching(locs, stg, long_term_goal, first_flag, whether_returning)
                if not locs.any():
                    return None
                # here is the room to room moving part
                locs, stg, long_term_goal = room_moving(locs, stg, long_term_goal, first_flag)
                if not locs.any():
                    return None
                locs, stg, long_term_goal, whether_returning = take_action(4, locs, first_flag)
                if not locs.any():
                    return None
                # after achieve new room, scan for new info, maybe it has already been scanned
                print('moved to another room')
            t_end = time()
            print('time cost {}'.format(t_end-t_start))
            while action_count < args.max_episode_length-1:
                locs, stg, long_term_goal, _ = take_action(0, locs, first_flag, [args.map_size_cm / 20, args.map_size_cm / 20])

            locs, stg, long_term_goal, _ = take_action(0, locs, False, [args.map_size_cm / 20, args.map_size_cm / 20])
            # take one more step to activate the new map
            #print('action_in total {}'.format(action_count))

        exploration(locs)



        #np.save('/home/airs/Downloads/ANS/Neural-SLAM/result_save/{}_result/my_step_list_detect.npy'.format(scene_name), step_list)
        #np.save('/home/airs/Downloads/ANS/Neural-SLAM/result_save/{}_result/my_cov_list_detect.npy'.format(scene_name), cov_ratio_list)

        # np.save('/home/airs/Downloads/ANS/Neural-SLAM/result_save/{}_result/my_step_list_current_real_door.npy'.format(scene_name), step_list)
        # np.save('/home/airs/Downloads/ANS/Neural-SLAM/result_save/{}_result/my_cov_list_current_real_door.npy'.format(scene_name), cov_ratio_list)
        # plt.close()

        """if action_count != 0:
            locs = np.array([6, 6, 0])
            while locs.any():
                print('action_ {}'.format(action_count))
                locs, _, _ = take_action(4, locs, False)"""
    """while keep_exploring:
        keystroke = input('input action command: ')#cv.waitKey(0)
        if keystroke == 'w':
            action_value = 2
            #locs, origins = take_action(action_value, locs, origins)
            locs = take_action(action_value, locs, first_flag)
            #first_flag = False
        elif keystroke == 'a':
            action_value = 0
            locs = take_action(action_value, locs, first_flag)
            #first_flag = False
        elif keystroke == 'd':
            action_value = 1
            locs = take_action(action_value, locs, first_flag)
            #first_flag = False
        elif keystroke == 'f':
            keep_exploring = False
        elif keystroke == 's':
            action_value = 4
            locs = take_action(action_value, locs, first_flag)
            first_flag = False"""

    # exit()


if __name__ == "__main__":
    main()

# save the origin take_action
"""        def take_action(action, locs):
        if action != 4:
            obs, rew, done, infos = envs.step(torch.tensor([action]))#
            #print(infos[0]['sensor_pose'])
            #print(locs)
            #locs = locs + infos[0]['sensor_pose']
            locs = pu.get_new_pose(locs, infos[0]['sensor_pose'])
            absolute_locs = locs + origins[0]

            r, c = absolute_locs[1], absolute_locs[0]
            loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                            int(c * 100.0 / args.map_resolution)]

            lmb[0] = get_local_map_boundaries((loc_r, loc_c),
                                              (local_w, local_h),
                                              (full_w, full_h))
            planner_pose_inputs[0, 3:] = lmb[0]
            origins[0] = [lmb[0][2] * args.map_resolution / 100.0,
                          lmb[0][0] * args.map_resolution / 100.0, 0.]
            locs = absolute_locs - origins[0]
            print('origin {}'.format(origins[0]))
            print('absolute_locs {}'.format(absolute_locs))
            print('locs {}'.format(locs))
            #goal_list = generate_12_parts(locs)
            #print(goal_list)
            angle = np.deg2rad(locs[2])
            dx = np.sin(angle)
            dy = np.cos(angle)
            obs_show = obs[0][0].cpu().detach().numpy()
            #obs_show = obs_show.transpose(2,1,0)
            gt_map = infos[0]['gt_map']#['gt_map']
            #gt_map = gt_map[::-1, ::-1]
            #gt_map = gt_map[::-1,:]  # flip vertically
            #gt_map = gt_map[:, ::-1]  # flip horizontally
            gt_map = gt_map.transpose()
            plt.clf()
            plt.subplot(1,2,1)
            plt.imshow(gt_map)
            plt.arrow(absolute_locs[1]*100/5, absolute_locs[0]*100/5,dx * 8, dy * (8 * 1.25),head_width=8, head_length=8 * 1.25,
                    length_includes_head=True, fc='Red', ec='Red', alpha=0.9)#absolute_locs[1]*100/5, absolute_locs[0]*100/5
            #plt.plot(origins[0][0]*100/5,origins[0][1]*100/5,'o',color='red')
            plt.plot(lmb[0][0], lmb[0][2], 'o', color='red')
            plt.plot(lmb[0][1], lmb[0][2], 'o', color='red')
            plt.plot(lmb[0][0], lmb[0][3], 'o', color='red')
            plt.plot(lmb[0][1], lmb[0][3], 'o', color='red')
            #for goal in goal_list:
            #    plt.plot(goal[0]+origins[0][1]*100/5,goal[1]+origins[0][0]*100/5,'o',color='yellow')
            plt.subplot(1,2,2)
            plt.imshow(obs_show, cmap='gray')
            plt.show()
        else:
            action = 1  # 12 turn_right makes up on scan motion
            for _ in range(12):
                obs, rew, done, infos = envs.step(torch.tensor([action]))  #
                # print(infos[0]['sensor_pose'])
                # print(locs)
                locs = pu.get_new_pose(locs, infos[0]['sensor_pose'])

                absolute_locs = locs + origins[0]
                r, c = absolute_locs[1], absolute_locs[0]
                loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                                int(c * 100.0 / args.map_resolution)]

                lmb[0] = get_local_map_boundaries((loc_r, loc_c),
                                                  (local_w, local_h),
                                                  (full_w, full_h))
                [gx1, gx2, gy1, gy2] = lmb[0]
                planner_pose_inputs[0, 3:] = lmb[0]
                planner_pose_inputs[0, :3] = absolute_locs
                origins[0] = [lmb[0][2] * args.map_resolution / 100.0,
                              lmb[0][0] * args.map_resolution / 100.0, 0.]
                locs = absolute_locs - origins[0]
                #print('origin {}'.format(origins[0]))
                #print('absolute_locs {}'.format(absolute_locs))
                #print('locs {}'.format(locs))

                # print(locs)
                angle = np.deg2rad(locs[2])
                dx = np.sin(angle)
                dy = np.cos(angle)
                obs_show = obs[0][0].cpu().detach().numpy()
            print('origin {}'.format(origins[0]))
            print('absolute_locs {}'.format(absolute_locs))
            print('locs {}'.format(locs))
            gt_map = infos[0]['pano_map']  # ['gt_map']
            gt_exp = infos[0]['pano_exp']
            #gt_map = gt_map.transpose()
            #gt_exp = gt_exp.transpose()

            #gt_map_local_grid = np.rint(gt_map[gy1:gy2, gx1:gx2])
            #gt_exp_local_grid = np.rint(gt_exp[gy1:gy2, gx1:gx2])
            gt_map_local_grid = np.rint(gt_map[gx1:gx2, gy1:gy2])
            gt_exp_local_grid = np.rint(gt_exp[gx1:gx2, gy1:gy2])

            #plt.imshow(gt_map_local_grid.transpose())
            #plt.plot(locs[1]*100/5, locs[0]*100/5, 'o', color = 'green')
            #plt.show()
            #exit()
            gt_map = gt_map.transpose()
            gt_exp = gt_exp.transpose()
            print(planner_pose_inputs)
            c, r = absolute_locs[:-1]
            start = [int(r * 100.0 / args.map_resolution - gx1),
                     int(c * 100.0 / args.map_resolution - gy1)]
            goal_list = generate_12_parts(locs)
            planner_pose_inputs_copy = planner_pose_inputs.copy()
            unexplorable_list = []
            list_x = []
            list_y = []
            for goal in goal_list:
                #goal = [int(302-origins[0][1]*100/5), int(237-origins[0][0]*100/5)]  # [x, y]
                #goal = [182, 117]
                print(goal)

                planner_inputs = [{} for e in range(num_scenes)]
                for e, p_input in enumerate(planner_inputs):
                    p_input['goal'] = goal
                    p_input['map_pred'] = gt_map_local_grid
                    p_input['exp_pred'] = gt_exp_local_grid
                    p_input['pose_pred'] = planner_pose_inputs[0]
                    p_input['mid_out'] = True
                loc_y = (planner_pose_inputs[0][0] - origins[0][0]) * 100 / 5
                loc_x = (planner_pose_inputs[0][1] - origins[0][1]) * 100 / 5
                total_dist = 0

                #list_x = []
                #list_y = []
                for _ in range(150):
                    output = envs.get_short_term_goal(planner_inputs)
                    #print(output)
                    dist = output[0][0].cpu().detach().numpy()
                    total_dist += dist
                    stg = output[0][1:].cpu().detach().numpy()
                    stg_x, stg_y = stg
                    #global_stg = np.array([stg_y, stg_x]) + np.array([origins[0][0] * 100 / 5, origins[0][1] * 100 / 5])
                    list_x.append(stg_x + origins[0][1] * 100 / 5)
                    list_y.append(stg_y + origins[0][0] * 100 / 5)
                    #print(total_dist)
                    if total_dist > explorable_threshold:
                        unexplorable_list.append(goal)
                        break

                    planner_pose_inputs[0,:2] = stg_y*5/100 + origins[0][0], stg_x*5/100 + origins[0][1]
                    for e, p_input in enumerate(planner_inputs):
                        p_input['goal'] = goal
                        p_input['map_pred'] = gt_map_local_grid
                        p_input['exp_pred'] = gt_exp_local_grid
                        p_input['pose_pred'] = planner_pose_inputs[0]
                        p_input['mid_out'] = True
                print(total_dist)
                if total_dist == 0.0:
                    print('goal in obstacle')
                    unexplorable_list.append(goal)
                planner_pose_inputs[0,:2] = planner_pose_inputs_copy[0,:2]

            #global_stg = np.array([stg_y, stg_x]) + np.array([origins[0][0]*100/5, origins[0][1]*100/5])
            #print(global_stg)
            #print(dist)
            #start = pu.threshold_poses(start, gt_map_local_grid.shape)
            #stg = envs._get_stg(gt_map_local_grid, gt_exp_local_grid, start, goal, [gx1, gx2, gy1, gy2])

            #gt_map = gt_map[::-1, :]  # filp vertically
            #gt_map = gt_map[::-1, ::-1]
            #exit()
            print(len(goal_list)-len(unexplorable_list))
            print(unexplorable_list)
            plt.clf()
            plt.subplot(1, 2, 1)
            plt.imshow(gt_map)
            plt.plot(lmb[0][0], lmb[0][2], 'o', color='green')
            plt.plot(lmb[0][1], lmb[0][2], 'o', color='green')
            plt.plot(lmb[0][0], lmb[0][3], 'o', color='green')
            plt.plot(lmb[0][1], lmb[0][3], 'o', color='green')
            plt.arrow(absolute_locs[1]*100/5, absolute_locs[0]*100/5,dx * 8, dy * (8 * 1.25),head_width=8, head_length=8 * 1.25,
                    length_includes_head=True, fc='Red', ec='Red', alpha=0.9)#absolute_locs[1]*100/5, absolute_locs[0]*100/5
            #plt.plot(global_stg[1], global_stg[0], 'o', color='blue')
            plt.plot(list_x, list_y, 'o', color='blue')
            for goal in goal_list:
                if goal not in unexplorable_list:
                    plt.plot(goal[0]+origins[0][1]*100/5, goal[1]+origins[0][0]*100/5, 'o', color='red')
            plt.subplot(1, 2, 2)
            plt.imshow(obs_show, cmap='gray')
            plt.show()
        return locs#, origins"""

# below version will crop the local map every time(but this might lead to wrong result)
"""    def take_action(action, locs):
        if action != 4:
            obs, rew, done, infos = envs.step(torch.tensor([action]))#
            #print(infos[0]['sensor_pose'])
            #print(locs)
            #locs = locs + infos[0]['sensor_pose']
            locs = pu.get_new_pose(locs, infos[0]['sensor_pose'])
            absolute_locs = locs + origins[0]

            r, c = absolute_locs[1], absolute_locs[0]
            loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                            int(c * 100.0 / args.map_resolution)]

            lmb[0] = get_local_map_boundaries((loc_r, loc_c),
                                              (local_w, local_h),
                                              (full_w, full_h))
            planner_pose_inputs[0, 3:] = lmb[0]
            origins[0] = [lmb[0][2] * args.map_resolution / 100.0,
                          lmb[0][0] * args.map_resolution / 100.0, 0.]
            locs = absolute_locs - origins[0]
            print('origin {}'.format(origins[0]))
            print('absolute_locs {}'.format(absolute_locs))
            print('locs {}'.format(locs))
            #goal_list = generate_12_parts(locs)
            #print(goal_list)
            angle = np.deg2rad(locs[2])
            dx = np.sin(angle)
            dy = np.cos(angle)
            obs_show = obs[0][0].cpu().detach().numpy()
            #obs_show = obs_show.transpose(2,1,0)
            gt_map = infos[0]['gt_map']#['gt_map']
            #gt_map = gt_map[::-1, ::-1]
            #gt_map = gt_map[::-1,:]  # flip vertically
            #gt_map = gt_map[:, ::-1]  # flip horizontally
            gt_map = gt_map.transpose()
            plt.clf()
            plt.subplot(1,2,1)
            plt.imshow(gt_map)
            plt.arrow(absolute_locs[1]*100/5, absolute_locs[0]*100/5,dx * 8, dy * (8 * 1.25),head_width=8, head_length=8 * 1.25,
                    length_includes_head=True, fc='Red', ec='Red', alpha=0.9)#absolute_locs[1]*100/5, absolute_locs[0]*100/5
            #plt.plot(origins[0][0]*100/5,origins[0][1]*100/5,'o',color='red')
            plt.plot(lmb[0][0], lmb[0][2], 'o', color='red')
            plt.plot(lmb[0][1], lmb[0][2], 'o', color='red')
            plt.plot(lmb[0][0], lmb[0][3], 'o', color='red')
            plt.plot(lmb[0][1], lmb[0][3], 'o', color='red')
            #for goal in goal_list:
            #    plt.plot(goal[0]+origins[0][1]*100/5,goal[1]+origins[0][0]*100/5,'o',color='yellow')
            plt.subplot(1,2,2)
            plt.imshow(obs_show, cmap='gray')
            plt.show()
        else:
            action = 1  # 12 turn_right makes up on scan motion
            for _ in range(12):
                obs, rew, done, infos = envs.step(torch.tensor([action]))  #
                # print(infos[0]['sensor_pose'])
                # print(locs)
                locs = pu.get_new_pose(locs, infos[0]['sensor_pose'])

                absolute_locs = locs + origins[0]
                r, c = absolute_locs[1], absolute_locs[0]
                loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                                int(c * 100.0 / args.map_resolution)]

                lmb[0] = get_local_map_boundaries((loc_r, loc_c),
                                                  (local_w, local_h),
                                                  (full_w, full_h))
                [gx1, gx2, gy1, gy2] = lmb[0]
                planner_pose_inputs[0, 3:] = lmb[0]
                planner_pose_inputs[0, :3] = absolute_locs
                origins[0] = [lmb[0][2] * args.map_resolution / 100.0,
                              lmb[0][0] * args.map_resolution / 100.0, 0.]
                locs = absolute_locs - origins[0]
                #print('origin {}'.format(origins[0]))
                #print('absolute_locs {}'.format(absolute_locs))
                #print('locs {}'.format(locs))

                # print(locs)
                angle = np.deg2rad(locs[2])
                dx = np.sin(angle)
                dy = np.cos(angle)
                obs_show = obs[0][0].cpu().detach().numpy()
            print('origin {}'.format(origins[0]))
            print('absolute_locs {}'.format(absolute_locs))
            print('locs {}'.format(locs))
            gt_map = infos[0]['pano_map']  # ['gt_map']
            gt_exp = infos[0]['pano_exp']
            #gt_map = gt_map.transpose()
            #gt_exp = gt_exp.transpose()

            #gt_map_local_grid = np.rint(gt_map[gy1:gy2, gx1:gx2])
            #gt_exp_local_grid = np.rint(gt_exp[gy1:gy2, gx1:gx2])
            gt_map_local_grid = np.rint(gt_map[gx1:gx2, gy1:gy2])
            gt_exp_local_grid = np.rint(gt_exp[gx1:gx2, gy1:gy2])

            #plt.imshow(gt_map_local_grid.transpose())
            #plt.plot(locs[1]*100/5, locs[0]*100/5, 'o', color = 'green')
            #plt.show()
            #exit()

            #print(planner_pose_inputs)
            c, r = absolute_locs[:-1]
            start = [int(r * 100.0 / args.map_resolution - gx1),
                     int(c * 100.0 / args.map_resolution - gy1)]
            goal_list = generate_12_parts(locs)
            #goal_list = [[68, 90], [68, 90]]
            planner_pose_inputs_copy = planner_pose_inputs.copy()
            origins_copy = origins.copy()
            absolute_locs_copy = absolute_locs.copy()
            lmb_copy = lmb.copy()
            gt_map_local_grid_copy = np.rint(gt_map[gx1:gx2, gy1:gy2])
            gt_exp_local_grid_copy = np.rint(gt_exp[gx1:gx2, gy1:gy2])

            unexplorable_list = []
            list_x = []
            list_y = []
            for goal in goal_list:
                #goal = [int(302-origins[0][1]*100/5), int(237-origins[0][0]*100/5)]  # [x, y]
                #goal = [182, 117]
                print('pre_goal{}'.format(goal))
                goal_copy = goal.copy()
                planner_inputs = [{} for e in range(num_scenes)]
                for e, p_input in enumerate(planner_inputs):
                    p_input['goal'] = goal
                    p_input['map_pred'] = gt_map_local_grid
                    p_input['exp_pred'] = gt_exp_local_grid
                    p_input['pose_pred'] = planner_pose_inputs[0]
                    p_input['mid_out'] = True
                loc_y = (planner_pose_inputs[0][0] - origins[0][0]) * 100 / 5
                loc_x = (planner_pose_inputs[0][1] - origins[0][1]) * 100 / 5
                total_dist = 0

                #list_x = []
                #list_y = []
                for _ in range(150):
                    output = envs.get_short_term_goal(planner_inputs)
                    #print(output)
                    dist = output[0][0].cpu().detach().numpy()
                    total_dist += dist
                    stg = output[0][1:].cpu().detach().numpy()
                    stg_x, stg_y = stg

                    #global_stg = np.array([stg_y, stg_x]) + np.array([origins[0][0] * 100 / 5, origins[0][1] * 100 / 5])
                    list_x.append(stg_x + origins[0][1] * 100 / 5)
                    list_y.append(stg_y + origins[0][0] * 100 / 5)
                    global_goal = [int(goal[0] + origins[0][0]*100/5), int(goal[1] + origins[0][1]*100/5)]
                    #print(total_dist)
                    if total_dist > explorable_threshold:
                        unexplorable_list.append(goal_copy)
                        break

                    planner_pose_inputs[0,:2] = stg_y*5/100 + origins[0][0], stg_x*5/100 + origins[0][1]
                    absolute_locs = planner_pose_inputs[0, :3]

                    r, c = absolute_locs[1], absolute_locs[0]
                    loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                                    int(c * 100.0 / args.map_resolution)]
                    #print('loc_r {}, loc_c {}'.format(loc_r, loc_c))
                    lmb[0] = get_local_map_boundaries((loc_r, loc_c),
                                                      (local_w, local_h),
                                                      (full_w, full_h))
                    [gx1, gx2, gy1, gy2] = lmb[0]
                    planner_pose_inputs[0, 3:] = lmb[0]
                    #planner_pose_inputs[0, :3] = absolute_locs

                    origins[0] = [lmb[0][2] * args.map_resolution / 100.0,
                                  lmb[0][0] * args.map_resolution / 100.0, 0.]
                    gt_map_local_grid = np.rint(gt_map[gx1:gx2, gy1:gy2])
                    gt_exp_local_grid = np.rint(gt_exp[gx1:gx2, gy1:gy2])
                    #locs = absolute_locs - origins[0]
                    goal = [int(global_goal[0] - origins[0][0]*100/5), int(global_goal[1] - origins[0][1]*100/5)]
                    #print('origins{}'.format(origins))
                    for e, p_input in enumerate(planner_inputs):
                        p_input['goal'] = goal
                        p_input['map_pred'] = gt_map_local_grid
                        p_input['exp_pred'] = gt_exp_local_grid
                        p_input['pose_pred'] = planner_pose_inputs[0]
                        p_input['mid_out'] = True
                print(total_dist)
                if total_dist == 0.0:
                    print('goal in obstacle')
                    unexplorable_list.append(goal_copy)
                #planner_pose_inputs[0,:] = planner_pose_inputs_copy[0,:]
                absolute_locs = absolute_locs_copy
                origins[:,:] = origins_copy[:,:]
                lmb[0] = lmb_copy[0]
                planner_pose_inputs[0, :3] = absolute_locs
                planner_pose_inputs[0, 3:] = lmb[0]
                [gx1, gx2, gy1, gy2] = lmb[0]
                gt_map_local_grid = gt_map_local_grid_copy
                gt_exp_local_grid = gt_exp_local_grid_copy

            #global_stg = np.array([stg_y, stg_x]) + np.array([origins[0][0]*100/5, origins[0][1]*100/5])
            #print(global_stg)
            #print(dist)
            #start = pu.threshold_poses(start, gt_map_local_grid.shape)
            #stg = envs._get_stg(gt_map_local_grid, gt_exp_local_grid, start, goal, [gx1, gx2, gy1, gy2])

            #gt_map = gt_map[::-1, :]  # filp vertically
            #gt_map = gt_map[::-1, ::-1]
            #exit()
            print(len(goal_list)-len(unexplorable_list))
            print(unexplorable_list)
            print(goal_list)
            gt_map = gt_map.transpose()
            gt_exp = gt_exp.transpose()
            plt.clf()
            plt.subplot(1, 2, 1)
            plt.imshow(gt_map)
            plt.plot(lmb[0][0], lmb[0][2], 'o', color='green')
            plt.plot(lmb[0][1], lmb[0][2], 'o', color='green')
            plt.plot(lmb[0][0], lmb[0][3], 'o', color='green')
            plt.plot(lmb[0][1], lmb[0][3], 'o', color='green')
            plt.arrow(absolute_locs[1]*100/5, absolute_locs[0]*100/5,dx * 8, dy * (8 * 1.25),head_width=8, head_length=8 * 1.25,
                    length_includes_head=True, fc='Red', ec='Red', alpha=0.9)#absolute_locs[1]*100/5, absolute_locs[0]*100/5
            #plt.plot(global_stg[1], global_stg[0], 'o', color='blue')
            plt.plot(list_x, list_y, 'o', color='blue')
            for goal in goal_list:
            #    if goal not in unexplorable_list:
                plt.plot(goal[0]+origins[0][1]*100/5, goal[1]+origins[0][0]*100/5, 'o', color='red')
            plt.subplot(1, 2, 2)
            plt.imshow(obs_show, cmap='gray')
            plt.show()
        return locs#, origins"""