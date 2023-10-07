import numpy as np
from treelib import Tree
from matplotlib import pyplot as plt
import math
from queue import Queue
import cv2


class Door_detection():
    def __init__(self, ini_list_x, ini_list_y):
        self.route = [{'route_idx':0, 'list_x':np.array(ini_list_x).reshape(-1, len(ini_list_x)),
                       'list_y':np.array(ini_list_y).reshape(-1, len(ini_list_y)), 'shortest_step':len(ini_list_x)}]
        self.route_num = len(self.route)
        self.route_idx = 1
        self.var_threshold = 1.
        self.kernel = np.ones((3, 3), np.uint8)  # for dilating the obs map
        
        # the variable below is used for obtaining the real door.
        self.door_candidate = []  # the candidate points of doors
        self.door_list = []  # the final list storing location of the door
        self.checking_size = 2  # this is half of the size of the square view range default 5
        self.door_detection_range = 20  # this is half of the size of the square view range default 15
        # checking mask is used for measuring the distance between the candidate door point with the nearest boundary
        self.checking_mask = self.generate_distance_mask(self.checking_size)  
        # door detection mask is used for measuring the distance between door point with all boundary in sight
        self.door_detection_mask = self.generate_distance_mask(self.door_detection_range)
        self.door_detection_circle = np.zeros_like(self.door_detection_mask, np.uint8)

        self.door_detection_circle = cv2.circle(self.door_detection_circle,
                                                (self.door_detection_range, self.door_detection_range),
                                                self.door_detection_range, 1, thickness = -1)

        # direction threshold is used for checking the angle between the door and the wall default 0.85
        self.direction_threshold = 0.7  # smaller, stricter
        # para_door_threshold is used for checking whether two doors are parallel
        self.para_door_threshold = 0.95
        # door distance threshold is used for limiting the distance between different two doors
        self.door_distance_threshold = 8 # 10
        self.door_distance_threshold_para = 25#40  # distance threshold for two paralle doors
        # door size threshold is the prior of the minimum size of the door
        self.door_size_threshold = 10
        # low limit and high limit are used for limiting the searching range within the cropped sight
        self.low_limit = 0
        self.high_limit = 2*self.door_detection_range
        # noisy point threshold is the minimum size of the obstacle that the door point belongs to
        self.noisy_point_threshold = 5
        # wall direction threshold is used for extracting multiple wall directions
        self.wall_direction_threshold = 0.4
        self.robot_current_loc = [ini_list_x[0], ini_list_y[0]]
        self.bot_near_range = 30


    def new_list(self, new_list_x, new_list_y):
        create_route = True
        point_1 = np.array([new_list_x[1], new_list_y[1]])
        for i in range(self.route_num):
            self.route_idx = i
            #print(self.route[self.route_idx]['list_x'].view().shape)
            point_exist = np.array([self.route[self.route_idx]['list_x'][0, 1],
                                    self.route[self.route_idx]['list_y'][0, 1]])
            dist = self.get_distance(point_1, point_exist)
            if dist < 1.5:  # belong to the same route
                create_route = False
                new_list_length = len(new_list_x)
                if new_list_length < self.route[self.route_idx]['shortest_step']:
                    self.route[self.route_idx]['shortest_step'] = new_list_length
                    self.route[self.route_idx]['list_x'] = \
                        np.row_stack((self.route[self.route_idx]['list_x'][:, :new_list_length],
                                     new_list_x))
                    self.route[self.route_idx]['list_y'] = \
                        np.row_stack((self.route[self.route_idx]['list_y'][:, :new_list_length],
                                     new_list_y))
                    #print(self.route[self.route_idx]['list_x'].shape)
                else:
                    new_list_length = self.route[self.route_idx]['shortest_step']
                    self.route[self.route_idx]['list_x'] = \
                        np.row_stack((self.route[self.route_idx]['list_x'],
                                      new_list_x[:new_list_length]))
                    self.route[self.route_idx]['list_y'] = \
                        np.row_stack((self.route[self.route_idx]['list_y'],
                                      new_list_y[:new_list_length]))
                break
        if create_route:
            self.route_num += 1
            self.route.append({'route_idx':self.route_num - 1, 'list_x':np.array(new_list_x).reshape(-1, len(new_list_x)),
                               'list_y':np.array(new_list_y).reshape(-1, len(new_list_x)), 'shortest_step':len(new_list_x)})
            #self.tree_idx = len(self.tree_node_count) - 1

    def get_distance(self, point_1, point_2):
        return ((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2) ** 0.5

    def cal_var(self):
        for i in range(self.route_num):
            self.route_idx = i
            var_x = np.var(self.route[self.route_idx]['list_x'], axis=0)
            var_y = np.var(self.route[self.route_idx]['list_y'], axis=0)
            avg_x = np.average(self.route[self.route_idx]['list_x'], axis=0)
            avg_y = np.average(self.route[self.route_idx]['list_y'], axis=0)
            var_total = var_x + var_y
            self.route[self.route_idx]['var'] = var_total
            self.route[self.route_idx]['sub_route_num'] = self.route[self.route_idx]['list_x'].shape[0]
            self.route[self.route_idx]['avg_x'] = avg_x
            self.route[self.route_idx]['avg_y'] = avg_y

    def get_door_point(self):
        self.cal_var()
        #print(self.route)
        door_list = []
        for i in range(self.route_num):
            self.route_idx = i
            if self.route[self.route_idx]['sub_route_num'] > 1:
                var = self.route[self.route_idx]['var']
                for j in range(self.route[self.route_idx]['shortest_step'] - 1):
                    if var[j] < self.var_threshold and var[j+1] > self.var_threshold:
                        door_list.append([int(self.route[self.route_idx]['avg_x'][j]),
                                          int(self.route[self.route_idx]['avg_y'][j])])
                        break
            else:  # only calculate the route that contains multiple sub_route
                continue
        self.door_candidate.extend(door_list)
        return door_list

    def reset(self, ini_list_x, ini_list_y):
        self.route.clear()
        self.route = [{'route_idx': 0, 'list_x': np.array(ini_list_x).reshape(-1, len(ini_list_x)),
                       'list_y': np.array(ini_list_y).reshape(-1, len(ini_list_y)), 'shortest_step': len(ini_list_x)}]
        self.route_num = len(self.route)
        self.route_idx = 1
        self.door_candidate.clear()

    def close_door(self, start, end, map_size):
        line_canvas = np.zeros((map_size, map_size), np.uint8)
        line_canvas = cv2.line(line_canvas, tuple(start), tuple(end), 1, 3)
        door_pixel_list = np.argwhere(line_canvas == 1).tolist()

        return door_pixel_list

    def get_waypoint(self, start, end, mid, gt_map, gt_exp, door_map):

        start = np.array(start)
        end = np.array(end)
        mid = np.array(mid)
        door_dir = end - start
        door_length = np.linalg.norm(door_dir)
        #print('door_dir {}'.format(door_dir))
        if door_dir[1] >= 0:
            vertical_dir = [1, -door_dir[0]/(door_dir[1] + 0.001)] #1 0
        else:
            vertical_dir = [1, -door_dir[0] / (door_dir[1] - 0.001)]
        #    vertical_dir = [0, 1]
        #print(vertical_dir)
        vertical_dir = vertical_dir/np.linalg.norm(vertical_dir)
        waypoint1 = (mid + 0.85*door_length*vertical_dir).astype(int).tolist()
        waypoint2 = (mid - 0.85*door_length*vertical_dir).astype(int).tolist()  # 0.7 is better, but failed in cantwell
        scale_factor = 1
        max_iter_num = 50
        while gt_exp[waypoint1[1], waypoint1[0]] != 1 or gt_map[waypoint1[1], waypoint1[0]] == 1 or door_map[waypoint1[1], waypoint1[0]] == 1:  #gt_map[waypoint1[1], waypoint1[0]] == 1 and
            waypoint1 = (mid + 0.9**scale_factor * door_length * vertical_dir).astype(int).tolist()
            if scale_factor > max_iter_num:
                waypoint1 = None
                break
            scale_factor += 1
        scale_factor = 1
        while gt_exp[waypoint2[1], waypoint2[0]] != 1 or gt_map[waypoint2[1], waypoint2[0]] == 1 or door_map[waypoint2[1], waypoint2[0]] == 1:  # gt_exp[waypoint1[1], waypoint1[0]] != 1 or gt_map[waypoint1[1], waypoint1[0]] == 1
            waypoint2 = (mid - 0.9**scale_factor * door_length * vertical_dir).astype(int).tolist()
            if scale_factor > max_iter_num:
                waypoint2 = None
                break
            scale_factor += 1
        return waypoint1, waypoint2

    def door_filter(self, global_obs_map, full_obs_map, gt_exp, bot_global_loc, bot_last_loc, external_door_list, use_12point=True, external_door_point=None):  # bot loc [x, y]
        self.door_list.clear()
        self.door_list = external_door_list.copy()  # the door list is given by the exploration.py now
        bot_mask = np.zeros((gt_exp.shape[0], gt_exp.shape[1]), np.uint8)
        bot_mask = cv2.circle(bot_mask, (round(bot_global_loc[0]), round(bot_global_loc[1])), self.bot_near_range,
                              1, -1)
        gt_exp_waypoint = gt_exp.copy()
        gt_exp_waypoint[bot_mask == 1] = 1

        global_obs_map = cv2.dilate(global_obs_map, self.kernel, iterations=1)
        global_obs_map = cv2.erode(global_obs_map, self.kernel, iterations=1)  # wipe out small hole and gap
        global_obs_map_with_door = global_obs_map.copy()
        full_obs_map_with_door = full_obs_map.copy()
        for door in self.door_list:
            start = door['start']
            end = door['end']
            door_pixel_list = self.close_door(start, end, global_obs_map_with_door.shape[1])
            for door_pixel in door_pixel_list:
                global_obs_map_with_door[door_pixel[0], door_pixel[1]] = 1
                full_obs_map_with_door[door_pixel[0], door_pixel[1]] = 1


        door_list_this_round = []
        door_candidate_this_round = []
        if use_12point:
            self.get_door_point()
        else:
            self.door_candidate.extend(external_door_point)
            door_candidate_this_round.extend(external_door_point)
        first_filter_out = []
        door_dict_copy = []
        for door in door_candidate_this_round:  # door is [x, y] and under global frame  originally in self.door_candidate
            # local_obs is the local obstacle map centered at candidate door point with the size of checking size,
            # it is used for checking whether the candidate point is far from the boundary
            local_obs = global_obs_map[door[1]-self.checking_size:door[1]+self.checking_size+1,
                        door[0]-self.checking_size:door[0]+self.checking_size+1]

            # filter out those points who are far from the boundary
            if local_obs.sum() == 0:
                print('remove door{}'.format(door))
                #self.door_candidate.remove(door)
                continue

            # find the closest boundary around the candidate point and move the point to the boundary
            if local_obs[self.checking_size, self.checking_size] == 0:
                dist_local_obs = local_obs * self.checking_mask
                dist_local_obs[dist_local_obs==0] = np.nan
                nearest_index = np.where(dist_local_obs == np.nanmin(dist_local_obs))
                door_2_obs = [nearest_index[1][0] - self.checking_size, nearest_index[0][0] - self.checking_size]
                # door_obs is the door point that moved to the nearest boundary, under global frame
                door_obs = [door[0]+door_2_obs[0], door[1]+door_2_obs[1]]
            else:
                door_obs = [door[0], door[1]]

            # door_detection_obs is the local obstacle map centered at the door_obs point with the size of
            # door detection range, it is used for detecting the corresponding door point
            door_detection_obs = \
                global_obs_map[
                            door_obs[1] - self.door_detection_range:door_obs[1] + self.door_detection_range + 1,
                            door_obs[0] - self.door_detection_range:door_obs[0] + self.door_detection_range + 1].copy()
            if door_detection_obs.shape[0] != 41 or door_detection_obs.shape[1] != 41:
                continue
            door_detection_obs = door_detection_obs*self.door_detection_circle  # convert into circle detection range
            """plt.ion()
            plt.imshow(door_detection_obs)
            plt.show()
            plt.pause(0.6)
            plt.ioff()
            plt.close()"""
            # next, we are going to remove all the obstacle points that are connected with the center point
            #print('connectivity check start')
            #np.save('bad_map.npy', door_detection_obs)
            # the wall_vec contains two elements: wall_vec and remove flag
            door_detection_obs, wall_vec = self.connectivity_check(door_detection_obs, [self.door_detection_range,
                                                                              self.door_detection_range])
            remove_flag = wall_vec[1]  # if remove flag is True, it means this door point might be noise
            if remove_flag:
                print('door {} removed for noise'.format(door_obs))
                continue
            wall_vec = wall_vec[0]
            #print('connectivity check done')
            corresponding_list = []
            while door_detection_obs.sum() != 0:
                ddo_copy = door_detection_obs.copy()
                range_map = ddo_copy*self.door_detection_mask
                range_map[range_map==0] = np.nan
                nearest_index = np.where(range_map == np.nanmin(range_map))  # maybe more than one results, take the 1st
                nearest_index = [nearest_index[0][0], nearest_index[1][0]]  #[y, x]
                nearest_index_diff = [nearest_index[1]-self.door_detection_range,
                                      nearest_index[0]-self.door_detection_range]
                nearest_index_global = [door_obs[0]+nearest_index_diff[0],
                                        door_obs[1]+nearest_index_diff[1]]  #[x, y]
                #print(nearest_index_global)
                door_detection_obs, obs_size = self.connectivity_check(door_detection_obs, nearest_index)
                #print('obs size {}'.format(obs_size))
                if obs_size < self.noisy_point_threshold:
                    continue
                # next we need to check the direction to filter out those doors parallel with the obstacle
                door_direction = np.array([nearest_index[1]-self.door_detection_range,
                                           nearest_index[0]-self.door_detection_range])  # [x,y]
                door_direction = door_direction/np.linalg.norm(door_direction)
                add_flag = True
                for wall_direction in wall_vec:
                    direction_check = np.dot(door_direction, wall_direction)
                    #print(direction_check)
                    if direction_check > self.direction_threshold:
                        add_flag = False
                        print('door {} {} removed for intersect with wall, direction check {}, wall vec {}'.format(
                            door_obs, nearest_index_global, direction_check, wall_direction))
                        break
                if add_flag:
                    corresponding_list.append([nearest_index_global, door_direction])
                #break
            door_dict = self.door_presentation(door_obs, corresponding_list)
            door_dict_copy.extend(door_dict)

            if len(self.door_list) != 0:  # finally, we filter out those doors with similar mid point,
                # door_size smaller than 5 and those intersect with obstacle

                for candidate_door in door_dict:
                    #print('door_dict_copy {}'.format(door_dict_copy))
                    add_flag = True
                    mid_candidate = candidate_door['mid']
                    door_direction = np.array(candidate_door['direction'])
                    start_candidate = candidate_door['start']
                    end_candidate = candidate_door['end']
                    next_point = np.array(start_candidate)+door_direction
                    door_obs_mask = np.zeros_like(global_obs_map, np.uint8)

                    bot_door_obs_mask = np.zeros_like(global_obs_map, np.uint8)
                    door_obs_mask = \
                        cv2.line(door_obs_mask, tuple(start_candidate), tuple(end_candidate), 1, thickness=1)
                    bot_door_obs_mask = \
                        cv2.line(bot_door_obs_mask, tuple(mid_candidate), tuple(bot_global_loc), 1, thickness=1)
                    """plt.subplot(1,2,1)
                    plt.imshow(bot_door_obs_mask)
                    plt.subplot(1,2,2)
                    plt.imshow(full_obs_map)
                    plt.show()"""
                    door_obs_intersect = door_obs_mask*full_obs_map_with_door#global_obs_map_with_door#may be full map is better
                    bot_door_obs_intersect = bot_door_obs_mask*full_obs_map_with_door#global_obs_map_with_door
                    size_candidate = self.get_distance(candidate_door['start'], candidate_door['end'])
                    check_door_list = []
                    check_door_list.extend(self.door_list)
                    check_door_list.extend(door_dict_copy)
                    check_door_list=list(check_door_list)
                    check_door_list.remove(candidate_door)

                    #print('check door list {}'.format(check_door_list))
                    if size_candidate > self.door_size_threshold:
                        for existed_door in check_door_list:#self.door_list
                            mid_existed = existed_door['mid']
                            door_door_obs_mask = np.zeros_like(global_obs_map, np.uint8)
                            door_door_obs_mask = \
                                cv2.line(door_door_obs_mask, tuple(mid_existed), tuple(mid_candidate), 1, thickness=1)
                            door_door_obs_intersect = door_door_obs_mask*full_obs_map
                            mid_distance = self.get_distance(mid_candidate, mid_existed)
                            direction_exist = np.array(existed_door['direction'])
                            direction_check = abs(np.dot(door_direction, direction_exist))
                            #print('mid dist {}'.format(mid_distance))
                            if direction_check < self.para_door_threshold:
                                if mid_distance < self.door_distance_threshold:
                                    add_flag = False
                                    print('door {} removed for too close door with {}'.format(candidate_door, existed_door))
                                    door_dict_copy.remove(candidate_door)
                                    break
                            else:
                                if mid_distance < self.door_distance_threshold_para and np.sum(door_door_obs_intersect) == 0:
                                    add_flag = False
                                    print('door {} removed for too close door para with {}'.format(candidate_door, existed_door))
                                    door_dict_copy.remove(candidate_door)
                                    break
                            if np.sum(bot_door_obs_intersect) > 0: # if not use gt_pano_map and global obs map with door, is 1,otherwise 0
                                add_flag = False
                                print('obs between bots and door {}, {}'.format(candidate_door, np.sum(bot_door_obs_intersect)))
                                door_dict_copy.remove(candidate_door)
                                break
                            if np.sum(door_obs_intersect) > 2:#global_obs_map[round(next_point[1]), round(next_point[0])] == 1:
                                # door intersect with obstacle will be removed
                                add_flag = False
                                print('door {} removed for intersect with obs, inter num {}'.format(candidate_door
                                                                                        ,np.sum(door_obs_intersect)))
                                print('next point {}'.format(next_point))
                                print('door direction {}'.format(door_direction))
                                door_dict_copy.remove(candidate_door)
                                break  # wait for improvement
                        if add_flag:
                            first_filter_out.append(candidate_door)
                            #self.door_list.append(candidate_door)
                    else:
                        print('removed door {} for too small size'.format(candidate_door))
                        door_dict_copy.remove(candidate_door)
            else:
                add_flag = True
                for candidate_door in door_dict:
                    size_candidate = self.get_distance(candidate_door['start'], candidate_door['end'])
                    check_door_list = []
                    check_door_list.extend(self.door_list)
                    check_door_list.extend(door_dict_copy)
                    check_door_list.remove(candidate_door)
                    add_flag = True
                    mid_candidate = candidate_door['mid']
                    if size_candidate > self.door_size_threshold:
                        for existed_door in check_door_list:
                            # add door distance here !!!!!!!!!!!!!!!!
                            mid_existed = existed_door['mid']
                            mid_distance = self.get_distance(mid_candidate, mid_existed)
                            if mid_distance < self.door_distance_threshold:
                                add_flag = False
                                print('door {} removed for too close door with {}'.format(candidate_door, existed_door))
                                door_dict_copy.remove(candidate_door)
                                break
                        if add_flag:
                            first_filter_out.append(candidate_door)
                        #self.door_list.append(candidate_door)
        print('first filter out {}'.format(first_filter_out))
        for candidate_door in first_filter_out:
            print('checking {} sec'.format(candidate_door))
            copy_ffo = first_filter_out.copy()
            copy_ffo.remove(candidate_door)
            add_flag = True
            mid_candidate = candidate_door['mid']
            start_candidate = candidate_door['start']
            door_direction = np.array(candidate_door['direction'])
            end_candidate = candidate_door['end']
            if len(copy_ffo) != 0:
                for other_door in copy_ffo:
                    other_start = other_door['start']
                    other_end = other_door['end']
                    print(other_door)
                    if self.cross_check(other_start, bot_global_loc, other_end, mid_candidate) < 0:
                        add_flag = False
                        print('door {} removed for sec order door'.format(candidate_door))
                        break
            door_map = np.zeros((gt_exp.shape[1], gt_exp.shape[1]), np.uint8)
            other_door_list = []  # other door list is the existed doors and this round's detected doors
            other_door_list.extend(self.door_list)
            other_door_list.extend(copy_ffo)
            for exist_door in other_door_list:
                door_map = cv2.line(door_map, tuple(exist_door['start']), tuple(exist_door['end']), 1,
                                    thickness=3)  # 3 is the same in frontier detection
            way_point_1, way_point_2 = self.get_waypoint(start_candidate, end_candidate, mid_candidate,
                                                         full_obs_map, gt_exp_waypoint, door_map)
            if way_point_1 is None or way_point_2 is None:
                add_flag = False
                print('removed door {} for can not find way point'.format(candidate_door))
                continue
            way_point_obs_mask = np.zeros((gt_exp.shape[1], gt_exp.shape[1]), np.uint8)
            way_point_obs_mask = cv2.line(way_point_obs_mask, tuple(np.round(way_point_1)),
                                          tuple(np.round(way_point_2)), 1, thickness = 1)
            way_point_obs_mask = way_point_obs_mask*full_obs_map

            if np.sum(way_point_obs_mask) > 0:
                add_flag = False
                print('removed door {} for obs between waypoints'.format(candidate_door))

            candidate_door['way_point_1'] = way_point_1
            candidate_door['way_point_2'] = way_point_2

            if add_flag:
                #self.door_list.append(candidate_door)
                door_list_this_round.append(candidate_door)
        """for door in door_list_this_round:
            for last_loc in bot_last_loc:
                if self.cross_check(door['start'], bot_global_loc, door['end'], last_loc) < 0:
                    print(self.get_distance(last_loc, door['mid']))
                    if self.get_distance(last_loc, door['mid']) <= 10:
                        door['cross_flag'] = True
                        print('door {} detected after crossing the door'.format(door))
                        break
                else:
                    door['cross_flag'] = False"""
        #np.save('door_list_door_detection.npy',self.door_list)
        return door_list_this_round, door_candidate_this_round#self.door_candidate

    def door_remove(self, door):
        try:
            self.door_list.remove(door)
        except:
            pass

    def door_presentation(self, door, corresponding_list):
        """

        Args:
            door:
            corresponding_list:

        Returns: presentation of each door
        {'start': start point, 'end': end point, 'mid': midd point of start and end}

        """
        tmp_list= []
        for corresponding in corresponding_list:
            direction = corresponding[1]
            corresponding = corresponding[0]
            tmp_list.append({'start': door, 'end': corresponding,
                             'mid': ((np.array(door)+np.array(corresponding))//2).tolist(),
                             'direction': direction.tolist(),
                             'cross_flag': False
                             })
        return tmp_list

    def connectivity_check(self, door_detect_obs, start_point):  #start point is [y, x]
        candidate_list = Queue()
        ddo_copy = door_detect_obs.copy()
        in_candidate_list = []
        connectivity_list = []
        processed_list = []
        candidate_list.put(start_point)
        in_candidate_list.append(start_point)
        #plt.imshow(door_detect_obs)
        #plt.show()
        while not candidate_list.empty():
            p = candidate_list.get()
            for i in range(-1, 2):
                for j in range(-1, 2):
                    w = [p[0] + i, p[1] + j]
                    if w[0] > self.high_limit or w[0] < self.low_limit \
                            or w[1] > self.high_limit or w[1] < self.low_limit:
                        continue
                    if door_detect_obs[w[0], w[1]] == 1 and w not in in_candidate_list and w not in processed_list:
                        candidate_list.put(w)
                        in_candidate_list.append(w)
                    elif door_detect_obs[w[0], w[1]] == 0 and w not in processed_list:
                        processed_list.append(w)
            in_candidate_list.remove(p)
            connectivity_list.append(p)
            processed_list.append(p)
        for i in connectivity_list:
            door_detect_obs[i[0], i[1]] = 0
        if start_point != [self.door_detection_range, self.door_detection_range]:
            return door_detect_obs, len(connectivity_list)
        else:
            return door_detect_obs, self.detect_wall_direction(connectivity_list, ddo_copy)

    def detect_wall_direction(self, connectivity_list, ddo):
        vector_list = []
        remove_flag = True
        if len(connectivity_list) >= self.noisy_point_threshold:
            remove_flag = False
            for i, connect_point in enumerate(connectivity_list):  # connect point is [y, x]
                if connect_point[0] == 0 or connect_point[0] == self.door_detection_range*2 or \
                        connect_point[1] == 0 or connect_point[1] == self.door_detection_range*2:
                    continue
                point_neighbor = ddo[connect_point[0]-1:connect_point[0]+2,
                                 connect_point[1]-1:connect_point[1]+2].copy()
                point_neighbor[0,0] = 1
                point_neighbor[0,2] = 1
                point_neighbor[1,1] = 1
                point_neighbor[2,0] = 1
                point_neighbor[2,2] = 1
                point_neighbor = np.abs(point_neighbor-1).sum()
                if point_neighbor != 0:
                    if self.get_distance(connect_point, [self.door_detection_range, self.door_detection_range]) > 10: # 5
                        candidate_vec = np.flip(np.array(connect_point)) - np.array([self.door_detection_range,
                                                                            self.door_detection_range])
                        candidate_vec = candidate_vec/np.linalg.norm(candidate_vec)
                        if len(vector_list) == 0:
                            vector_list.append(candidate_vec)
                        else:
                            add_vec_flag = True
                            for existed_vec in vector_list:
                                dot_product = np.dot(existed_vec, candidate_vec)
                                if dot_product > self.wall_direction_threshold:
                                    add_vec_flag = False
                            if add_vec_flag:
                                vector_list.append(candidate_vec)
        return vector_list, remove_flag

    def generate_distance_mask(self, size):
        full_size = 2*size + 1
        center = [size, size]
        mask = np.zeros((full_size, full_size))
        for i in range(full_size):
            for j in range(full_size):
                mask[i, j] = math.sqrt((i - center[0])**2 + (j - center[1])**2)
        return mask

    def cross_check(self, start_point1, start_point2, end_point1, end_point2):
        # start_point1 is the start of other door, start_point2 is the current robot location

        vec_1 = np.array(start_point2) - np.array(start_point1)
        vec_2 = np.array(end_point2) - np.array(start_point1)
        vec_line = np.array(end_point1) - np.array(start_point1)
        check_result = np.dot(np.cross(vec_1, vec_line), np.cross(vec_2, vec_line))

        return np.sign(check_result)

        """self.tree = Tree()
        self.tree.create_node('0', '0', data={'node_type': 'root', 'value': 1, 'loc': np.array([
            ini_list_x[0], ini_list_y[0]]), 'angle_value': None, 'node_name': '0'})  # create the root
        self.tree_idx = 0  # indicate the idx of subtree
        self.tree_node_count = [0]  # count the number of nodes within subtree
        for j in range(ini_list_x.shape[0] - 1):
            j = j + 1  # the first term is the agent's location, so we start from 1
            if j != ini_list_x.shape[0]-1:  # the last term is the leave node
                pre_point = np.array([ini_list_x[j - 1], ini_list_y[j - 1]])
                current_point = np.array([ini_list_x[j], ini_list_y[j]])
                next_point = np.array([ini_list_x[j + 1], ini_list_y[j + 1]])
                angle_value = self.get_angle(pre_point, current_point, next_point)
                if j == 1:
                    self.tree_node_count[self.tree_idx] += 1
                    self.tree.create_node(
                        '{}_{}'.format(self.tree_idx, self.tree_node_count[self.tree_idx]),
                        '{}_{}'.format(self.tree_idx, self.tree_node_count[self.tree_idx]),
                        parent='{}'.format(0),
                        data={'node_type': 'child_node', 'value': 1, 'loc': current_point, 'angle_value': angle_value,
                              'node_name': '{}_{}'.format(self.tree_idx, self.tree_node_count[self.tree_idx])}
                    )
                else:
                    self.tree_node_count[self.tree_idx] += 1
                    self.tree.create_node(
                        '{}_{}'.format(self.tree_idx, self.tree_node_count[self.tree_idx]),
                        '{}_{}'.format(self.tree_idx, self.tree_node_count[self.tree_idx]),
                        parent='{}_{}'.format(self.tree_idx, self.tree_node_count[self.tree_idx]-1),
                        data={'node_type': 'child_node', 'value': 1, 'loc': current_point, 'angle_value': angle_value,
                              'node_name': '{}_{}'.format(self.tree_idx, self.tree_node_count[self.tree_idx])}
                    )
                #print(self.get_distacne(current_point, pre_point))
            else:
                self.tree_node_count[self.tree_idx] += 1
                current_point = np.array([ini_list_x[j], ini_list_y[j]])
                self.tree.create_node(
                    '{}_{}'.format(self.tree_idx, self.tree_node_count[self.tree_idx]),
                    '{}_{}'.format(self.tree_idx, self.tree_node_count[self.tree_idx]),
                    parent='{}_{}'.format(self.tree_idx, self.tree_node_count[self.tree_idx] - 1),
                    data={'node_type': 'leaf_node', 'value': 1, 'loc': current_point, 'angle_value': None,
                          'node_name': '{}_{}'.format(self.tree_idx, self.tree_node_count[self.tree_idx])}
                )  # the leaf node doesn't have angle value
        self.tree.show()

    def new_tree(self, new_list_x, new_list_y):
        create_tree = True
        point_1 = np.array([new_list_x[1], new_list_y[1]])
        for i in range(len(self.tree_node_count)):
            self.tree_idx = i
            point_exist = self.tree.nodes['{}_{}'.format(i, 1)].data['loc']
            dist = self.get_distacne(point_1, point_exist)
            if dist < 1.5:  # belong to the same subtree
                create_tree = False
                break
        if create_tree:
            self.tree_node_count.append(0)
            self.tree_idx = len(self.tree_node_count) - 1
        return create_tree

    def get_distacne(self, point_1, point_2):
        return ((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2) ** 0.5

    def get_angle(self, pre_point, current_point, next_point):
        mod_1 = self.get_distacne(current_point, pre_point)
        mod_2 = self.get_distacne(next_point, current_point)
        vec_1 = current_point - pre_point
        vec_2 = current_point - next_point
        dot_product = np.dot(vec_1, vec_2)
        return dot_product/(mod_1*mod_2)

    def add_new_list(self, new_list_x, new_list_y):
        create_tree = self.new_tree(new_list_x, new_list_y)  # first we check the necessity to create a new tree
        if not create_tree:  # which means we add the list to existing tree
            for j in range(new_list_x.shape[0] - 1):
                pass  # TO BE CONTINUE"""

    """def add_new_list(self, new_list_x, new_list_y):
        self.tree_idx += 1
        for j in range(new_list_x.shape[0] - 1):
            j = j + 1  # the first term is the agent's location, so we start from 1
            add_node = True
            if j != new_list_x.shape[0]-1:  # the last term is the leaf node
                pre_point = np.array([new_list_x[j - 1], new_list_y[j - 1]])
                current_point = np.array([new_list_x[j], new_list_y[j]])
                next_point = np.array([new_list_x[j + 1], new_list_y[j + 1]])
                angle_value = self.get_angle(pre_point, current_point, next_point)
                for node in self.tree.all_nodes_itr():
                    corresponding_node = node.data
                    corresponding_node_name = corresponding_node['node_name']
                    corresponding_node_loc = corresponding_node['loc']
                    corresponding_node_ang = corresponding_node['angle_value']
                    corresponding_node_value = corresponding_node['value']
                    dist = self.get_distacne(current_point, corresponding_node_loc)
                    #print(corresponding_node)
                    if dist <= 3:
                        corresponding_node_value += 1
                        self.tree.update_node('{}'.format(j),
                            data={'node_type': 'child_node', 'value': corresponding_node_value,
                                  'loc': corresponding_node_loc, 'angle_value': corresponding_node_ang,
                                  'node_name': corresponding_node_name})
                        add_node = False
                        break
                if add_node:
                    self.tree.create_node(
                        '{}'.format(self.tree_idx),
                        '{}'.format(self.tree_idx),
                        parent='{}'.format(self.tree_idx - 1),
                        data={'node_type': 'child_node', 'value': 1, 'loc': current_point,
                              'angle_value': angle_value, 'node_name': self.tree_idx}
                    )
                    self.tree_idx += 1"""



if __name__ == '__main__':
    list_x = np.load('list_x{}.npy'.format(0))
    list_y = np.load('list_y{}.npy'.format(0))
    door_detect = Door_detection(list_x, list_y)
    c = np.zeros((480, 480))
    for i in range(list_x.shape[0]):
        c[int(list_x[i]), int(list_y[i])] = 1
    for i in range(11):
        i = i+1
        new_list_x = np.load('list_x{}.npy'.format(i))
        new_list_y = np.load('list_y{}.npy'.format(i))
        door_detect.new_list(new_list_x, new_list_y)
        for j in range(new_list_x.shape[0]):
            c[int(new_list_x[j]), int(new_list_y[j])] = 1
    door = door_detect.get_door_point()
    #print(door)
    #c[239, 260] = 0
    plt.imshow(c)
    for i in range(len(door)):
        plt.plot(door[i][1], door[i][0], 'o', color='red')
    plt.show()
