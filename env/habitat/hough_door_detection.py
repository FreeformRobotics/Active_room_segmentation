import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from queue import Queue
from sklearn.cluster import DBSCAN, KMeans

gomez_distance = 3

def nothing(x):  # 滑动条的回调函数
    pass

def hough_detection(src):
    #src = cv2.imread('/home/airs/Downloads/ANS/Neural-SLAM/paper_fig/depth5.png',0)
    src = src.astype(np.uint8)
    line_mask = np.zeros((src.shape[0], src.shape[1]), np.uint8)

    #src = src[:,:,::-1]
    #srcBlur = cv2.GaussianBlur(src, (3, 3), 0)
    #gray = cv2.cvtColor(srcBlur, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(src, 50, 150, apertureSize=3)  # 50, 150
    #WindowName = 'Approx'  # 窗口名
    #cv2.namedWindow(WindowName, cv2.WINDOW_AUTOSIZE)  # 建立空窗口

    #cv2.createTrackbar('threshold', WindowName, 0, 60, nothing)  # 创建滑动条

    #while (1):
    img = src.copy()
    #threshold = 100 + 2 * cv2.getTrackbarPos('threshold', WindowName)  # 获取滑动条值
    threshold = 50

    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)

    try:
        for line in lines:
            rho = line[0][0]
            theta = line[0][1]
            if not (np.rad2deg(theta) < 10 or np.rad2deg(theta) >170):
                continue
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            #cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.line(line_mask, (x1, y1), (x2, y2), 1, 5)
    except:
        pass

    #    cv2.imshow(WindowName, img)
    #    k = cv2.waitKey(1) & 0xFF
    #    if k == 27:
    #        break
    #cv2.destroyAllWindows()
    #img = img[:,:,::-1]

    return line_mask#img


def generate_parts(agent_pose):
    devided_num = 360*2
    dist = 6
    x = agent_pose[1]  # x and y unit is m and presented in local frame
    y = agent_pose[0]
    angle = agent_pose[2]  # the angle unit is degree
    angle_list = [np.deg2rad(angle + i * 360//devided_num) for i in range(devided_num)]  # in the list, the unit becomes rad
    goal_list = []
    for content in angle_list:
        dx = dist * np.cos(content)  # 3 * np.cos(content)
        dy = dist * np.sin(content)  # 3 * np.sin(content)
        goal_list.append([round((x + dx) * 100 / 5), round((y + dy) * 100 / 5)])
    return goal_list

def generate_parts_gomez(agent_pose):
    devided_num = 360*2 # 360*2
    dist = gomez_distance
    x = agent_pose[1]  # x and y unit is m and presented in local frame
    y = agent_pose[0]
    angle = agent_pose[2]  # the angle unit is degree
    angle_list = [np.deg2rad(angle + i * 360//devided_num) for i in range(devided_num)]  # in the list, the unit becomes rad
    goal_list = []
    for content in angle_list:
        dx = dist * np.cos(content)  # 3 * np.cos(content)
        dy = dist * np.sin(content)  # 3 * np.sin(content)
        goal_list.append([round((x + dx) * 100 / 5), round((y + dy) * 100 / 5)])
    return goal_list


def generate_distance_mask(size, agent_pose):
    center = [agent_pose[0]*100/5, agent_pose[1]*100/5]
    mask = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            mask[i, j] = math.sqrt((i - center[0])**2 + (j - center[1])**2)
    return mask


def convert_2_laser(obs_map, exp_map, agent_pose): # agent_post in global frame and m [y,x,o]
    bot_near_range = 30
    gap_threshold = 15  #default 20
    map = np.ones((exp_map.shape[0], exp_map.shape[0])) * 0.5
    goal_list = generate_parts(agent_pose)
    bot_mask = np.zeros((exp_map.shape[0], exp_map.shape[0]), np.uint8)
    bot_mask = cv2.circle(bot_mask, (round(agent_pose[1]*100/5), round(agent_pose[0]*100/5)), bot_near_range, 1, -1)
    dist_mask = generate_distance_mask(exp_map.shape[0], agent_pose)
    exp_map += bot_mask
    map[exp_map != 0] = 0
    map[obs_map != 0] = 1
    laser_list = []
    laser_dist_list = []
    bot_out_of_map_flag = False

    for goal in goal_list:
        line_mask = np.zeros((exp_map.shape[0], exp_map.shape[0]), np.uint8)
        cv2.line(line_mask, (round(agent_pose[1]*100/5), round(agent_pose[0]*100/5)), tuple(goal), color=1, thickness=1)
        grid_state_list = map[line_mask == 1]
        grid_dist_list = dist_mask[line_mask == 1]
        #print('len of gdl {}'.format(len(grid_dist_list)))
        #print('len of gsl {}'.format(len(grid_state_list)))

        if len(grid_dist_list) != 0:
            grid_dist_list, grid_state_list = zip(*sorted(zip(grid_dist_list, grid_state_list)))

            idx = 0
            dist_on_line = dist_mask*line_mask

            for i in range(len(grid_state_list)):
                idx = i
                if grid_state_list[i] != 0:
                    break
            stop_dist = grid_dist_list[idx]

            grid_idx = np.where(dist_on_line == stop_dist)
            #print(grid_idx[0][0])
            laser_list.append((grid_idx[1][0], grid_idx[0][0]))  #  [x, y]
            laser_dist_list.append(stop_dist)
        else:
            bot_out_of_map_flag = True
    potential_door_list=[]
    diff_list = []
    for i in range(len(laser_dist_list)):
        if i == len(laser_dist_list) - 1:
            j = 0
        else:
            j = i+1
        diff_list.append(abs(laser_dist_list[i] - laser_dist_list[j]))
        if abs(laser_dist_list[i] - laser_dist_list[j]) > gap_threshold:
            if laser_dist_list[i]<laser_dist_list[j]:
                potential_door_list.append(laser_list[i])
            else:
                potential_door_list.append(laser_list[j])
    """#plt.subplot(1,2,1)
    plt.imshow(map)
    for grid_idx in laser_list:
        if grid_idx in potential_door_list:
            plt.plot(grid_idx[0], grid_idx[1], 'o', color='red')
        else:
            plt.plot(grid_idx[0], grid_idx[1], 'o', color='green')
    #plt.subplot(1,2,2)
    #plt.plot(diff_list)
    #plt.imshow(dist_on_line)
    plt.show()"""
    if bot_out_of_map_flag:
        potential_door_list.clear()
    return potential_door_list, laser_list


def convert_2_laser_gomez(obs_map, exp_map, agent_pose, gt_door_map): # agent_post in global frame and m [y,x,o]
    bot_near_range = 30
    gap_threshold = 20 #16  #default 20 16
    map = np.ones((exp_map.shape[0], exp_map.shape[0])) * 0.5
    kernel = np.ones((3, 3), np.uint8)
    exp_map = cv2.dilate(exp_map, kernel, iterations=1)
    goal_list = generate_parts_gomez(agent_pose)
    bot_mask = np.zeros((exp_map.shape[0], exp_map.shape[0]), np.uint8)
    bot_mask = cv2.circle(bot_mask, (round(agent_pose[1]*100/5), round(agent_pose[0]*100/5)), bot_near_range, 1, -1)
    dist_mask = generate_distance_mask(exp_map.shape[0], agent_pose)
    exp_map += bot_mask
    map[exp_map != 0] = 0
    map[obs_map != 0] = 1
    laser_list = []
    cov_area_list = []
    laser_dist_list = []
    full_length_list = []
    full_length_list_not_change = []
    #laser_state_list = []
    #bot_out_of_map_flag = False

    for scan_idx, goal in enumerate(goal_list):
        line_mask = np.zeros((exp_map.shape[0], exp_map.shape[0]), np.uint8)
        cv2.line(line_mask, (round(agent_pose[1]*100/5), round(agent_pose[0]*100/5)), tuple(goal), color=1, thickness=1)
        grid_state_list = map[line_mask == 1]
        grid_dist_list = dist_mask[line_mask == 1]
        #print('len of gdl {}'.format(len(grid_dist_list)))
        #print('len of gsl {}'.format(len(grid_state_list)))


        if len(grid_dist_list) != 0:
            grid_dist_list, grid_state_list = zip(*sorted(zip(grid_dist_list, grid_state_list)))

            idx = 0
            dist_on_line = dist_mask*line_mask

            for i in range(len(grid_state_list)):
                idx = i
                if grid_state_list[i] != 0:
                    break
            stop_dist = grid_dist_list[idx]

            grid_idx = np.where(dist_on_line == stop_dist)
            #print(grid_idx[0][0])
            laser_list.append([grid_idx[1][0], grid_idx[0][0]])  #  [x, y]
            laser_dist_list.append(stop_dist)

            if stop_dist >=gomez_distance*20-1 and stop_dist <=gomez_distance*20+1 and grid_state_list[idx] != 1:
                full_length_list.append([[grid_idx[1][0], grid_idx[0][0]], scan_idx])
                #full_length_list_not_change.append([[grid_idx[1][0], grid_idx[0][0]], scan_idx])
                #laser_state_list.append(grid_state_list[idx])
        #else:
        #    bot_out_of_map_flag = True

    potential_door_list=[]
    #diff_list = []
    door_flag = False
    for i in range(len(laser_dist_list)):

        if i == len(laser_dist_list) - 1:
            j = 0
        else:
            j = i + 1
        # diff_list.append(abs(laser_dist_list[i] - laser_dist_list[j]))
        if door_flag:
            if abs(laser_dist_list[i] - door_length) < gap_threshold:

                door_flag = False
            potential_door_list[-1].append([laser_list[i], i])
                # break
            # try:
            #    full_length_list.remove(laser_list[i])
            # except:
            #    pass
        # diff_list.append(abs(laser_dist_list[i] - laser_dist_list[j]))
        if abs(laser_dist_list[i] - laser_dist_list[j]) > gap_threshold and not door_flag:
            door_flag = True
            potential_door_list.append([])
            if laser_dist_list[i] < laser_dist_list[j]:
                potential_door_list[-1].append([laser_list[i], i])
                door_length = laser_dist_list[i]
            else:
                potential_door_list[-1].append([laser_list[j], j])
                door_length = laser_dist_list[j]

    door_flag = False
    for i in range(len(laser_dist_list)-1,-1,-1):

        if i == 0:
            j = len(laser_dist_list) - 1
        else:
            j = i-1
        #diff_list.append(abs(laser_dist_list[i] - laser_dist_list[j]))
        if door_flag:
            if abs(laser_dist_list[i]-door_length) < gap_threshold:
                door_flag = False
            potential_door_list[-1].append([laser_list[i], i])


                #break
            #try:
            #    full_length_list.remove(laser_list[i])
            #except:
            #    pass
        #diff_list.append(abs(laser_dist_list[i] - laser_dist_list[j]))
        if abs(laser_dist_list[i] - laser_dist_list[j]) > gap_threshold and not door_flag:
            door_flag = True
            potential_door_list.append([])
            if laser_dist_list[i]<laser_dist_list[j]:
                potential_door_list[-1].append([laser_list[i], i])
                door_length = laser_dist_list[i]
            else:
                potential_door_list[-1].append([laser_list[j], j])
                door_length = laser_dist_list[j]

    """for i in range(len(laser_dist_list)):

        if i == len(laser_dist_list) - 1:
            j = 0
        else:
            j = i+1
        #diff_list.append(abs(laser_dist_list[i] - laser_dist_list[j]))
        if door_flag:
            if abs(laser_dist_list[i]-door_length) > gap_threshold:
                potential_door_list.append(laser_list[i])
                try:
                    full_length_list.remove(laser_list[i])
                except:
                    pass
            else:
                door_flag = False
                #break
            #try:
            #    full_length_list.remove(laser_list[i])
            #except:
            #    pass
        #diff_list.append(abs(laser_dist_list[i] - laser_dist_list[j]))
        if abs(laser_dist_list[i] - laser_dist_list[j]) > gap_threshold and not door_flag:
            door_flag = True
            if laser_dist_list[i]<laser_dist_list[j]:
                potential_door_list.append(laser_list[j])
                door_length = laser_dist_list[i]
            else:
                potential_door_list.append(laser_list[i])
                door_length = laser_dist_list[j]
"""
    #queue_door = Queue
    combined_door_list = []
    combined_list = []
    processed_idx = []

    for i in range(len(potential_door_list)):
        if i not in processed_idx:
            processed_idx.append(i)
        else:
            continue
        content = potential_door_list[i]
        combined_list.extend(content)
        for j in range(i+1, len(potential_door_list)):
            tmp_list = get_intersection(potential_door_list[j], content)
            if len(tmp_list) > 0:
                processed_idx.append(j)
                combined_list.extend(potential_door_list[j])

        combined_door_list.append(combine_list(combined_list))

        combined_list.clear()

    result_door_list = []  # format like [[mid_point, info_size], [....]]

    for door in combined_door_list:

        if np.sum(gt_door_map[door[0][0][1] - 2:door[0][0][1] + 3, door[0][0][0] - 2:door[0][0][0] + 3]) == 0 \
                or np.sum(
            gt_door_map[door[-1][0][1] - 2:door[-1][0][1] + 3, door[-1][0][0] - 2:door[-1][0][0] + 3]) == 0:  # or  -2 +3
            print('removed door since vertical line not around the door frame!!!!!!!!!')
            continue

        mid_point = [(door[0][0][0] + door[-1][0][0]) // 2, (door[-1][0][1] + door[0][0][1]) // 2]
        door_grid_list = []
        for door_grid in door:
            door_grid_list.append(door_grid[1])

        #--------
        if 0 in door_grid_list and 719 in door_grid_list:
            gap_position = 0
            # gap_flag = False
            for idx_gap in range(len(door_grid_list) - 1):
                if door_grid_list[idx_gap] - door_grid_list[idx_gap + 1] == -1:
                    continue
                else:
                    gap_position = idx_gap
                    # gap_flag = True
                    break
            for idx_f in range(gap_position):
                door_grid_list[idx_f] += 720
        f_point_idx = np.round(np.median(door_grid_list))
        if f_point_idx > 719:
            f_point_idx -= 720
        #--------

        #f_point_idx = np.round(np.median(door_grid_list))

        f_point = None
        for door_grid in door:
            if f_point_idx == door_grid[1]:
                dist = \
                    np.sqrt((door_grid[0][0] - round(agent_pose[1] * 100 / 5)) ** 2 + (
                                door_grid[0][1] - round(agent_pose[0] * 100 / 5)) ** 2)
                print('dist {}'.format(dist))
                if dist >= gomez_distance*20-1:  # and dist <= 61
                    f_point = door_grid[0]
                    print('chosen {} as f_point'.format(f_point))
                    break


        geo_size = np.sqrt((door[0][0][0] - door[-1][0][0]) ** 2 + (door[-1][0][1] - door[0][0][1]) ** 2)
        info_size = len(door) - 2
        if geo_size >= 0.8 * 100 / 5 and geo_size <= 2.4 * 100 / 5:
            # geo_size >= 0.8 * 100 / 5 and geo_size <= 1.2 * 100 / 5 or geo_size >= 1.6 * 100 / 5 and geo_size <= 2.4 * 100 / 5
            # geo_size >= 0.8 * 100 / 5 and geo_size <= 2.4 * 100 / 5
            result_door_list.append([mid_point, info_size, f_point])
            for door_grid in door:
                try:
                    full_length_list.remove(door_grid)
                except:
                    pass
        else:
            print('removed for incorrect size {}!!!!!!!!!'.format(geo_size*5/100))


    #for content in potential_door_list:
        #if len(content[0]) < 10:
        #    potential_door_list.remove(content)
    #print(len(potential_door_list))
    full_length_list_coordinate = []
    for content in full_length_list:
        full_length_list_coordinate.append(content[0])
    frontier_list = []
    frontier_idx_list = []
    if len(full_length_list) != 0:
        result = DBSCAN(eps=10, min_samples=10).fit_predict(full_length_list_coordinate)  # eps=10, min_samples=10
        result_list = result.tolist()
        while -1 in result_list:  # -1 means noise
            result_list.remove(-1)
        result_set = set(result_list)
        frontier_num = len(result_set)
        #f_np = np.array(full_length_list)
        for i in range(frontier_num):
            coordinate_list = []
            idx_list = get_index1(result_list, i)
            full_length_list_not_change.append([])
            for idx_f in idx_list:
                coordinate_list.append(full_length_list[idx_f][1])
                full_length_list_not_change[-1].append(full_length_list[idx_f][0])
            if 0 in coordinate_list and 719 in coordinate_list:
                gap_position = 0
                #gap_flag = False
                for idx_gap in range(len(coordinate_list)-1):
                    if coordinate_list[idx_gap] - coordinate_list[idx_gap+1] == -1:
                        continue
                    else:
                        gap_position = idx_gap
                        #gap_flag = True
                        break
                for idx_f in range(gap_position):
                    coordinate_list[idx_f] += 720
            mid_idx = np.round(np.median(coordinate_list))
            if mid_idx > 719:
                mid_idx -= 720
            frontier_idx_list.append([mid_idx, len(idx_list)])  # np.round(np.median(coordinate_list))


    for idx_f in frontier_idx_list:
        f_size = idx_f[1]
        idx_f = idx_f[0]
        for content in full_length_list:
            if content[1] == idx_f:
                frontier_list.append([content[0], f_size])
                break

        """for i in range(frontier_num):
            frontier = np.sum(f_np[result == i], axis=0) / result_list.count(i)
            frontier_list.append(frontier)"""


    #if bot_out_of_map_flag:
    #    potential_door_list.clear()
    for content in result_door_list:
        if result_door_list.count(content) > 1:
            for _ in range(result_door_list.count(content) - 1):
                result_door_list.remove(content)
    return result_door_list, laser_list, frontier_list, combined_door_list, full_length_list_coordinate#full_length_list_not_change#potential_door_list

def combine_list(target_list):
    idx_set = set()
    for content in target_list:
        idx_set.add(content[1])
    idx_list = list(idx_set)
    idx_list.sort()
    output = []
    for idx in idx_list:
        for content in target_list:
            if idx == content[1]:
                output.append(content)
                break
    return output

def get_intersection(list1, list2):
    set1 = set()
    set2 = set()
    for content in list1:
        set1.add(content[1])
    for content in list2:
        set2.add(content[1])
    intesetction = set1&set2
    return list(intesetction)

def get_index1(lst=None, item=''):
     return [index for (index,value) in enumerate(lst) if value == item]

if __name__ == '__main__':
    for i in range(12):
        img = np.load('/home/airs/Downloads/ANS/Neural-SLAM/paper_fig/depth{}.npy'.format(i))
        img_ = hough_detection(img)
        plt.subplot(1,2,1)
        plt.imshow(img)
        plt.subplot(1,2,2)
        plt.imshow(img_)
        plt.show()
    """obs_map = np.load('scan_map.npy').astype(np.uint8)
    exp_map = np.load('scan_exp.npy').astype(np.uint8)
    #map = np.load('scan_map.npy')

    agent_pose = [12, 12, 0]
    goal_list = generate_parts(agent_pose)

    convert_2_laser(obs_map, exp_map, agent_pose)"""
    """plt.imshow(map)
    for i in goal_list:
        plt.plot(i[0], i[1], 'o', color='red')
    
    plt.show()"""