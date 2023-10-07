import numpy as np
from matplotlib import pyplot as plt
from queue import Queue
import cv2
import time
from sklearn.cluster import DBSCAN, KMeans
from arguments import get_args
args = get_args()


class Frontier_detection():
    def __init__(self, map_size):
        self.map_size = map_size
        self.q_m = Queue()
        self.q_f = Queue()
        self.kernel = np.ones((3, 3), np.uint8)  # for dilation
        self.new_frontier = []
        self.MCL = []
        self.MOL = []
        self.FOL = []
        self.FCL = []
        self.new_frontier_save = []
        self.bot_mask = np.zeros((map_size, map_size), np.uint8)
        self.door_gird = []  # set up the door_grid list to make sure that the door_grid will not be eroded by scanning
        #MCL: map-close-list
        #MOL: map-open-list
        #FOL: frontier-open-list
        #FCL: frontier-close-list
        self.bot_near_range = 30
        self.first_round_flag = True
        self.map = np.ones((self.map_size, self.map_size)) * 0.5

    def frontier_detection(self, start_point, current_pose, obs_map, exp_map, lmb, close_door_list, laser_list = None):  # the current_pose is under global frame[y, x]
        self.new_frontier_save.clear()
        self.MCL.clear()
        self.MOL.clear()
        self.FCL.clear()
        self.FOL.clear()
        self.new_frontier.clear()

        lmb = lmb.astype(int).tolist()
        self.obs_map = obs_map.copy()
        self.exp_map = exp_map.copy()


        rob_pose = current_pose.astype(int).tolist()
        # we need to cover the area around the bot for detection
        if laser_list:  # when explore use laser_list when used in topo check use circle mask
            for laser_point in laser_list:
                self.bot_mask = cv2.line(self.bot_mask, (rob_pose[1], rob_pose[0]), tuple(laser_point), color=1, thickness=2)
            #plt.clf()
            #plt.imshow(self.bot_mask)
            #plt.show()
        #else:
            #if self.first_round_flag:
            # comment this if, we filling the bot's surrounding every scan otherwise only the first time
            #self.bot_mask = cv2.circle(self.bot_mask, (rob_pose[1], rob_pose[0]), self.bot_near_range, 1, -1)
        self.exp_map += self.bot_mask
        self.bot_mask = np.zeros((self.map_size, self.map_size), np.uint8)
        self.first_round_flag = False


        self.obs_map = cv2.dilate(self.obs_map, self.kernel, iterations=1)
        #self.obs_map = cv2.erode(self.obs_map, self.kernel, iterations=1)
        self.exp_map = cv2.erode(self.exp_map, self.kernel, iterations=1)

        self.map[self.exp_map != 0] = 0
        self.map[self.obs_map != 0] = 1



        current_pose = start_point.astype(int)
        #print('bot current map status {}'.format(self.map[current_pose[0], current_pose[1]]))
        bot_local_map = self.map[current_pose[0]-1:current_pose[0]+2, current_pose[1]-1:current_pose[1]+2]
        if bot_local_map.shape[0] == 3 and bot_local_map.shape[1] == 3:
            if self.map[current_pose[0], current_pose[1]] != 0:
                try:
                    new_start_offset = np.argwhere(bot_local_map == 0)[0]-np.array([1,1])
                    current_pose += new_start_offset
                except:
                    pass
        current_pose = current_pose.tolist()
        if current_pose[0] <0:
            current_pose[0] = 0
        if current_pose[0] >= exp_map.shape[0]:
            current_pose[0] = exp_map.shape[0]-1
        if current_pose[1] <0:
            current_pose[1] = 0
        if current_pose[1] >= exp_map.shape[0]:
            current_pose[1] = exp_map.shape[0]-1
        # ------------------------------------------------------------------
        # this part is used for closing the door while detecting the frontier
        self.door_gird.clear()
        if len(close_door_list) > 0:
            for door in close_door_list:
                start = door['start']
                end = door['end']
                door_pixel_list = self.close_door(start, end)
                for door_pixel in door_pixel_list:
                    self.map[door_pixel[0], door_pixel[1]] = 0.1
                    if [door_pixel[0], door_pixel[1]] not in self.door_gird:
                        self.door_gird.append([door_pixel[0], door_pixel[1]])
                """for y in range(min(start[1], end[1]), max(end[1], start[1])):
                    for x in range(min(start[0], end[0]), max(end[0], start[0])):
                        self.map[y,x] = 0.1
                        if [y, x] not in self.door_gird:
                            self.door_gird.append([y,x])"""
                        #self.add_to_list([y, x], 'MCL')
        for door_gird in self.door_gird:
            self.map[door_gird[0], door_gird[1]] = 0.1
            self.add_to_list(door_gird, 'MCL')
        """close_door = []
        for y in range(235,260):
            for x in range(230,240):
                self.map[y, x] = 0.1
                close_door.append([y,x])
        self.MCL.extend(close_door)"""
        # ------------------------------------------------------------------
        #self.exp_map = cv2.dilate(self.exp_map, self.kernel, iterations=1)
        #self.exp_map = cv2.erode(self.exp_map, self.kernel, iterations=1)
        #self.map = self.exp_map + self.obs_map
        self.q_m.queue.clear()
        self.q_m.put(current_pose)
        self.add_to_list(current_pose, 'MOL')
        t1 = time.time()
        while not self.q_m.empty():
            p = self.q_m.get()
            if p in self.MCL or self.map[p[0], p[1]] == 1:# or self.map[p[0], p[1]] == 1
                # this is one difference with the original WFD that we ignore the obstacles so that we can constrain the
                # frontier in one room by closing the door.
                # we do not ignore the obs anymore since 2023 3.26 since it makes the search abort when bot
                # is close to obs
                continue
            if self.frontier_point(p):
                self.q_f.queue.clear()
                self.new_frontier.clear()
                self.add_to_list(p, 'FOL')
                self.q_f.put(p)
                while not self.q_f.empty():
                    q = self.q_f.get()
                    if q in self.MCL or q in self.FCL:
                        continue
                    if self.frontier_point(q):
                        self.new_frontier.append(q)
                        for i in range(-1, 2):
                            for j in range(-1, 2):
                                if i == 0 and j == 0:
                                    continue
                                w = [q[0]-i, q[1]-j]
                                if w not in self.FOL and w not in self.FCL and w not in self.MCL:
                                    self.q_f.put(w)
                                    self.add_to_list(w, 'FOL')
                    self.add_to_list(q, 'FCL')
                self.new_frontier_save.extend(self.new_frontier)
                for data in self.new_frontier:
                    self.add_to_list(data, 'MCL')
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if i == 0 and j == 0:
                        continue
                    v = [p[0]-i, p[1]-j]
                    if v not in self.MCL and v not in self.MOL and self.openspace_neighbor(v):
                        self.q_m.put(v)
                        self.add_to_list(v, 'MOL')
            self.add_to_list(p, 'MCL')
        #print(self.new_frontier)
        #print('frontier searching time {}'.format(time.time()-t1))
        """plt.ion()
        plt.imshow(self.map)
        for f in self.new_frontier_save:
            plt.plot(f[1], f[0], 'o', color='red')
        plt.show()
        plt.pause(1)
        plt.ioff()
        plt.close()"""

        """plt.clf()
        plt.imshow(self.map)
        for i in self.MCL:
            plt.plot(i[1], i[0], 'o')
        # plt.plot(way_point_1[1], way_point_1[0], 'o', color = 'blue')
        # plt.plot(way_point_2[1], way_point_2[0], 'o', color='purple')
        plt.show()"""
        f_list, info_gain_list = self.frontier_cluster(self.new_frontier_save, self.exp_map)
        return f_list, info_gain_list, self.map, self.MCL, self.door_gird

    def close_door(self, start, end):
        line_canvas = np.zeros((self.exp_map.shape[0], self.exp_map.shape[1]), np.uint8)
        line_canvas = cv2.line(line_canvas, tuple(start), tuple(end), 1, 3)
        door_pixel_list = np.argwhere(line_canvas == 1).tolist()

        return door_pixel_list

    def add_to_list(self, data, list_name):
        if list_name == 'MCL':
            if data in self.MCL:
                pass
            else:
                self.MCL.append(data)
            if data in self.MOL:
                self.MOL.remove(data)
            if data in self.FOL:
                self.FOL.remove(data)
            if data in self.FCL:
                self.FCL.remove(data)
        if list_name == 'MOL':
            if data in self.MOL:
                pass
            else:
                self.MOL.append(data)
            if data in self.MCL:
                self.MCL.remove(data)
            if data in self.FOL:
                self.FOL.remove(data)
            if data in self.FCL:
                self.FCL.remove(data)
        if list_name == 'FCL':
            if data in self.FCL:
                pass
            else:
                self.FCL.append(data)
            if data in self.MOL:
                self.MOL.remove(data)
            if data in self.FOL:
                self.FOL.remove(data)
            if data in self.MOL:
                self.MOL.remove(data)
        if list_name == 'FOL':
            if data in self.FOL:
                pass
            else:
                self.FOL.append(data)
            if data in self.MOL:
                self.MOL.remove(data)
            if data in self.MCL:
                self.MCL.remove(data)
            if data in self.FCL:
                self.FCL.remove(data)

    def openspace_neighbor(self, pose):
        neighbor = self.map[pose[0] - 1:pose[0] + 2, pose[1] - 1:pose[1] + 2]
        if neighbor.shape[0] !=3 or neighbor.shape[1] !=3:
            return False
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    continue
                else:
                    if neighbor[i, j] == 0:
                        return True
        return False

    def frontier_point(self, pose):
        neighbor = self.map[pose[0]-1:pose[0]+2, pose[1]-1:pose[1]+2]
        if neighbor.shape[0] != 3 or neighbor.shape[1] != 3:
            return False
        if neighbor[1,1] != 0.5:
            return False
        for i in range(neighbor.shape[0]):
            for j in range(neighbor.shape[1]):
                if i ==1 and j ==1:
                    continue
                else:
                    if neighbor[i, j] == 0:
                        return True
        return False

    def frontier_cluster(self, f_list, exp_map):

        if len(f_list) == 0:
            return [], []
        exp_map_ = np.zeros_like(exp_map)
        exp_map_[exp_map == 0] = 1
        result = DBSCAN(eps=10, min_samples=15).fit_predict(f_list)  # 15 can success, but find better, 20
        result_list = result.tolist()
        while -1 in result_list:  # -1 means noise
            result_list.remove(-1)
        result_set = set(result_list)
        frontier_num = len(result_set)
        f_np = np.array(f_list)
        frontier_list = []
        info_gain_list = []
        for i in range(frontier_num):
            frontier = np.sum(f_np[result == i], axis=0) / result_list.count(i)
            frontier_list.append(frontier)
            bot_detection_circle = np.zeros((exp_map.shape[1], exp_map.shape[1]), np.uint8)
            bot_detection_circle = cv2.circle(bot_detection_circle, (round(frontier[1]), round(frontier[0])),
                                              args.vision_range, 1, -1)
            info_map = bot_detection_circle*exp_map_
            #info_gain_list.append(np.sum(info_map))
            info_gain_list.append(result_list.count(i))
        return frontier_list, info_gain_list


if __name__ == '__main__':
    gt_map = np.load('gt_map.npy')
    gt_exp = np.load('gt_exp.npy')
    frontier_detector = Frontier_detection()
    current_pose = np.array([240, 240])
    lmb = np.array([120, 360, 120, 360])
    close_door_list = [{'start': [238, 255], 'end': [228, 244], 'mid': [233, 249]}]
    f_list, map = frontier_detector.frontier_detection(current_pose, gt_map, gt_exp, lmb, close_door_list)
    """result = DBSCAN(eps=10, min_samples=2).fit_predict(f_list)
    result_list = result.tolist()
    while -1 in result_list:  # -1 means noise
        result_list.remove(-1)
    result_set = set(result_list)
    frontier_num = len(result_set)
    f_np = np.array(f_list)
    frontier_list = []
    for i in range(frontier_num):
        frontier = np.sum(f_np[result == i], axis=0)/result_list.count(i)
        frontier_list.append(frontier)"""


    """plt.imshow(map)
    for f in f_list:  # f is [y, x]
        plt.plot(f[1], f[0], 'o', color='lawngreen')
    plt.plot(current_pose[1], current_pose[0], 'o', color = 'pink')
    plt.show()"""


# frontier detection class store
"""class Frontier_detection():
    def __init__(self):
        self.q_m = Queue()
        self.q_f = Queue()
        self.kernel = np.ones((3, 3), np.uint8)  # for dilation
        self.map = np.ones((480, 480))*0.5
        self.new_frontier = []
        self.MCL = []
        self.MOL = []
        self.FOL = []
        self.FCL = []
        self.new_frontier_save = []
        self.bot_mask = np.load('aroundbot_mask.npy')
        #MCL: map-close-list
        #MOL: map-open-list
        #FOL: frontier-open-list
        #FCL: frontier-close-list

    def frontier_detection(self, current_pose, obs_map, exp_map, lmb):  # the current_pose is under global frame[y, x]
        self.obs_map = obs_map.copy()
        self.exp_map = exp_map.copy()


        # we need to cover the area around the bot for detection
        self.exp_map[lmb[2]:lmb[3], lmb[0]:lmb[1]] += self.bot_mask

        self.obs_map = cv2.dilate(self.obs_map, self.kernel, iterations=1)
        #self.exp_map = cv2.dilate(self.exp_map, self.kernel, iterations=1)

        self.map[self.exp_map != 0] = 0
        self.map[self.obs_map != 0] = 1


        # ------------------------------------------------------------------
        # this part is used for closing the door while detecting the frontier
        close_door = []
        for y in range(235,260):
            for x in range(230,260):
                self.map[y, x] = 0.1
                close_door.append([y,x])
        self.MCL.extend(close_door)
        # ------------------------------------------------------------------

        #self.exp_map = cv2.dilate(self.exp_map, self.kernel, iterations=1)
        #self.exp_map = cv2.erode(self.exp_map, self.kernel, iterations=1)
        #self.map = self.exp_map + self.obs_map
        self.q_m.queue.clear()
        self.q_m.put(current_pose)
        self.add_to_list(current_pose, 'MOL')
        t1 = time.time()
        while not self.q_m.empty():
            p = self.q_m.get()
            if p in self.MCL:
                continue
            if self.frontier_point(p):
                self.q_f.queue.clear()
                self.new_frontier.clear()
                self.add_to_list(p, 'FOL')
                self.q_f.put(p)
                while not self.q_f.empty():
                    q = self.q_f.get()
                    if q in self.MCL or q in self.FCL:
                        continue
                    if self.frontier_point(q):
                        self.new_frontier.append(q)
                        for i in range(-1, 2):
                            for j in range(-1, 2):
                                if i == 0 and j == 0:
                                    continue
                                w = [q[0]-i, q[1]-j]
                                if w not in self.FOL and w not in self.FCL and w not in self.MCL:
                                    self.q_f.put(w)
                                    self.add_to_list(w, 'FOL')
                    self.add_to_list(q, 'FCL')
                self.new_frontier_save.extend(self.new_frontier)
                for data in self.new_frontier:
                    self.add_to_list(data, 'MCL')
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if i == 0 and j == 0:
                        continue
                    v = [p[0]-i, p[1]-j]
                    if v not in self.MCL and v not in self.MOL and self.openspace_neighbor(v):
                        self.q_m.put(v)
                        self.add_to_list(v, 'MOL')
            self.add_to_list(p, 'MCL')
        print(time.time()-t1)
        return self.new_frontier_save, self.map

    def add_to_list(self, data, list_name):
        if list_name == 'MCL':
            if data in self.MCL:
                pass
            else:
                self.MCL.append(data)
            if data in self.MOL:
                self.MOL.remove(data)
            if data in self.FOL:
                self.FOL.remove(data)
            if data in self.FCL:
                self.FCL.remove(data)
        if list_name == 'MOL':
            if data in self.MOL:
                pass
            else:
                self.MOL.append(data)
            if data in self.MCL:
                self.MCL.remove(data)
            if data in self.FOL:
                self.FOL.remove(data)
            if data in self.FCL:
                self.FCL.remove(data)
        if list_name == 'FCL':
            if data in self.FCL:
                pass
            else:
                self.FCL.append(data)
            if data in self.MOL:
                self.MOL.remove(data)
            if data in self.FOL:
                self.FOL.remove(data)
            if data in self.MOL:
                self.MOL.remove(data)
        if list_name == 'FOL':
            if data in self.FOL:
                pass
            else:
                self.FOL.append(data)
            if data in self.MOL:
                self.MOL.remove(data)
            if data in self.MCL:
                self.MCL.remove(data)
            if data in self.FCL:
                self.FCL.remove(data)

    def openspace_neighbor(self, pose):
        neighbor = self.map[pose[0] - 1:pose[0] + 2, pose[1] - 1:pose[1] + 2]
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    continue
                else:
                    if neighbor[i, j] == 0:
                        return True
        return False

    def frontier_point(self, pose):
        neighbor = self.map[pose[0]-1:pose[0]+2, pose[1]-1:pose[1]+2]
        if neighbor[1,1] != 0.5:
            return False
        for i in range(3):
            for j in range(3):
                if i ==1 and j ==1:
                    continue
                else:
                    if neighbor[i, j] == 0:
                        return True
        return False"""