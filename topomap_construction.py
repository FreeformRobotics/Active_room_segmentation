import numpy as np
import igraph as ig
from matplotlib import pyplot as plt
from frontier_detection import Frontier_detection
from arguments import get_args
import cv2
args = get_args()

class Topomap_construction():
    def __init__(self):
        #self.vertice = []
        self.g = ig.Graph(n=1, directed=True)  # the robot initial position is one room node
        self.current_node_id = 0
        self.v_num = self.g.vcount()
        #self.g.vs['room_idx'] = [0]
        self.g.es['way_point'] = []  # shown then global point of going through one door, one door has two waypoints
        self.g.vs['room_status'] = ['exploring']
        self.g.vs['room_entry'] = [[]]
        self.g.vs['room_exp'] = [[]]
        # record whether a room is explored, have three status: exploring, explored, unexplored
        self.in_room_point = None
        self.use_topo = False
        self.frontier_detector = Frontier_detection(args.map_size_cm // args.map_resolution)
        self.bot_near_range = 30

    def same_node_check(self, room_exp_list, detected_door_list): # map will be deleted soon
        target_node_idx_list = []

        door_remove = None
        combine_flag = False
        if self.v_num > 1:
            for i in range(self.v_num):
                if i != self.current_node_id:
                    for node_entry in self.g.vs[i]['room_entry']:
                        room_entry = node_entry.copy()
                        room_entry.reverse()  # [y, x], global, pix
                        if room_entry in room_exp_list and self.g.vs[i]['room_status'] == 'unexplored':
                            print('TRUE {} {}'.format(room_entry, self.current_node_id))
                            combine_flag = True
                            target_node_idx_list.append(i)
                            break
            print('same node check result {} {}'.format(target_node_idx_list, self.current_node_id))
            if combine_flag:

                for target_node_idx in target_node_idx_list:  # combine the target node with the current node
                    self_loop = False
                    for node_entry in self.g.vs[target_node_idx]['room_entry']:
                        # we move the room entry from target to current node
                        if not node_entry in self.g.vs[self.current_node_id]['room_entry']:
                            self.g.vs[self.current_node_id]['room_entry'].append(node_entry)
                    # move the way point from target edge to new established edge
                    connected_inedge_list = self.g.incident(target_node_idx, mode='in')
                    connected_outedge_list = self.g.incident(target_node_idx, mode='out')
                    # first we build in edge and move the waypoint to it
                    for inedge_idx in connected_inedge_list:
                        in_node = self.get_connected_v(inedge_idx, target_node_idx)
                        if in_node == self.current_node_id:
                            self_loop = True
                            self.g.vs[self.current_node_id]['room_entry'].remove(self.g.es[inedge_idx]['way_point'])
                            #break
                            continue
                        self.g.add_edges([[in_node, self.current_node_id]])
                        self.g.es[self.g.ecount()-1]['way_point'] = self.g.es[inedge_idx]['way_point']
                    # second we build out edge and move the waypoint to it

                    for outedge_idx in connected_outedge_list:
                        out_node = self.get_connected_v(outedge_idx, target_node_idx)
                        if out_node == self.current_node_id:
                            self_loop = True
                            self.g.vs[self.current_node_id]['room_entry'].remove(self.g.es[outedge_idx]['way_point'])
                            #break
                            continue
                        self.g.add_edges([[self.current_node_id, out_node]])
                        self.g.es[self.g.ecount()-1]['way_point'] = self.g.es[outedge_idx]['way_point']
                    if self_loop:
                        loop_waypoint_list = []  # remove those doors do not enclose and found during further exploration
                        for node_entry in self.g.vs[target_node_idx]['room_entry']:
                            loop_waypoint_list.append(node_entry)
                        for door in detected_door_list:
                            if door['way_point_1'] == loop_waypoint_list[0] or door['way_point_2'] == loop_waypoint_list[0]:
                                door_remove = door
                                break
                if door_remove:
                    detected_door_list.remove(door_remove)
                    print('door {} removed for not enclose after steps of exp'.format(door_remove))
                # finally we remove the target node
                self.g.delete_vertices(target_node_idx_list)
                self.update_topo(room_exp_list)
        return detected_door_list

    def update_topo(self, room_exp_list):
        """
        here, we need to update properties of topo like current node id or v_num after combine the nodes
        Args:
            room_exp_list:

        Returns:

        """
        self.v_num = self.g.vcount()
        for i in range(self.v_num):
                for node_entry in self.g.vs[i]['room_entry']:
                    room_entry = node_entry.copy()
                    room_entry.reverse()  # [y, x], global, pix
                    if room_entry in room_exp_list:
                        self.current_node_id = i
                        print('relocate current node id to {}'.format(self.current_node_id))
                        break

    def check_topomap(self, door_list, detected_door_list, room_exp_list, bot_loc, gt_map, gt_exp, lmb, door_grid, scene_idx, scene_name, laser_list, demo = False):
        # room exp list is the frontier MCL start from bot's current loc and current_node_exp_list starts from node's entry
        bot_loc_xy = list(bot_loc)

        bot_loc_xy.reverse()
        out_of_current = False  # a flag shown whether the robot is not in the current node
        door_remove_list = []
        for door in door_list:
            way_point_1 = door['way_point_1'].copy()
            way_point_1.reverse()
            way_point_2 = door['way_point_2'].copy()
            way_point_2.reverse()

            """plt.clf()
            plt.ion()
            plt.imshow(gt_map)
            for i in room_exp_list:
                plt.plot(i[1], i[0], 'o')
            plt.plot(way_point_1[1], way_point_1[0], 'o', color = 'cyan')
            plt.plot(bot_loc_xy[0], bot_loc_xy[1], 'o', color='red')
            #plt.plot(way_point_2[1], way_point_2[0], 'o', color='purple')
            #plt.show()
            plt.pause(10)
            plt.ioff()
            plt.close()"""
            if way_point_1 in room_exp_list:
                door['p1_same_side'] = True
            else:
                if way_point_2 in room_exp_list:
                    door['p1_same_side'] = False
                else:
                    print('p1 comfirmed by is same side func')
                    if self.if_same_side(bot_loc_xy, door['way_point_1'], door['start'], door['end']):
                        door['p1_same_side'] = True
                    else:
                        door['p1_same_side'] = False
            if way_point_1 in room_exp_list and way_point_2 in room_exp_list:  # means the door is not enclosed
                door_remove_list.append(door)
                print('door {} removed for not enclosed'.format(door))
        for door in door_remove_list:
            door_list.remove(door)
        out_of_current_list = []
        # we make it a list rather than a flag since some false cases, so once entry is in list, means not out
        for node_entry in self.g.vs[self.current_node_id]['room_entry']:
            room_entry = node_entry.copy()
            room_entry.reverse()

            if room_entry not in room_exp_list:
                out_of_current_list.append(False)
            else:
                out_of_current_list.append(True)
                #print(room_exp_list)
                #exit()
                #break
        if sum(out_of_current_list) > 0 or self.v_num == 1:
            # means at least one entry is in exp list, and the first node is always in the room
            out_of_current = False
        else:
            out_of_current = True
            print('bot out of room!!!')
        try:
            room_entry = self.g.vs[self.current_node_id]['room_entry'][0].copy()
            room_entry.reverse()
            _, _, _, current_node_exp_list, door_grid = self.frontier_detector.frontier_detection(np.array(room_entry),
                                                                                    np.array(bot_loc),
                                                                                    np.rint(gt_map), np.rint(gt_exp),
                                                                                    lmb,
                                                                                    detected_door_list,
                                                                                                  laser_list)
        except:
            current_node_exp_list = room_exp_list
        #print(current_node_exp_list)
        self.g.vs[self.current_node_id]['room_exp'] = current_node_exp_list.copy()
        for door_pix in door_grid:
            self.g.vs[self.current_node_id]['room_exp'].remove(door_pix)
        tmp_show_map = np.zeros_like(gt_map)
        for i in range(self.g.vcount()):
            room_label = i+1
            room_exp_list_ = self.g.vs[i]['room_exp']
            try:
                for j in room_exp_list_:
                    tmp_show_map[j[0], j[1]] = room_label
            except:
                pass
        #np.save('/home/airs/Downloads/ANS/Neural-SLAM/topo_mapping_result/Hambleton/{}/topo_room_map_Hambleton.npy'.format(scene_idx), tmp_show_map)#scene_idx
        """np.save(
            '/home/airs/Downloads/ANS/Neural-SLAM/8m_add_exp/accuracy/Ours/{}/{}/topo_room_map_{}.npy'.format(scene_name,
                scene_idx, scene_name), tmp_show_map)"""

        """plt.ion()
        plt.imshow(tmp_show_map)
        plt.show()
        plt.pause(1)
        plt.ioff()
        plt.close()"""

        if out_of_current:  # means that the detected door are not in current node

            for door in door_list:
                way_point_1 = door['way_point_1'].copy()
                way_point_2 = door['way_point_2'].copy()
                way_point_1.reverse()
                way_point_2.reverse()
                if way_point_1 in current_node_exp_list or way_point_2 in current_node_exp_list:
                    door['cross_flag'] = True
                    break  # we assume bot can only cross one door at a time
        if not demo:
            return door_list, door_remove_list
        else:
            return door_list, door_remove_list, tmp_show_map

    def get_graph(self):
        return self.g

    def add_room(self, door_list, current_location, gt_map, gt_exp, lmb):  # current_location is [x, y], unit is pix
        #print('add room door list {}'.format(door_list))
        bot_mask = np.zeros((gt_exp.shape[0], gt_exp.shape[1]), np.uint8)
        bot_mask = cv2.circle(bot_mask, (round(current_location[0]), round(current_location[1])), self.bot_near_range, 1, -1)
        gt_exp_waypoint = gt_exp.copy()
        gt_exp_waypoint[bot_mask == 1] = 1
        cross_flag_exist = False
        cross_idx = 0
        crossed_door = None

        # first, we need to check whether there exists a cross door, and we assume that there only exist one cross door
        for cross_door_idx in range(len(door_list)):
            if door_list[cross_door_idx]['cross_flag']: # after finding the cross flag door, we need to put it the first place of the list
                cross_flag_exist = True
                crossed_door = door_list[cross_door_idx]
                break
        if cross_flag_exist:
            door_list.remove(crossed_door)
            door_list.insert(0, crossed_door)

        if not self.use_topo:
            return_flag = False
            #current_location.reverse()  # convert to [x, y]
            new_v = len(door_list)
            #self.vertice.extend(vertice)
            self.g.add_vertices(new_v)
            cross_node_idx = 0
            tmp_current_node_idx_store = self.current_node_id
            if cross_flag_exist:
                cross_node_idx = self.v_num

            for i in range(self.v_num, self.g.vcount()):  # get the new added nodes' id
                # a pair of edges represents a door
                if i != self.v_num and cross_flag_exist:
                    self.current_node_id = cross_node_idx
                self.g.vs[i]['room_status'] = 'unexplored'
                self.g.vs[i]['room_entry'] = []
                door_idx = i - self.v_num
                door = door_list[door_idx]
                start = door['start']
                end = door['end']
                mid = door['mid']
                """way_point_1 = np.array([start[0], end[1]])
                way_point_2 = np.array([end[0], start[1]])
                way_point_1 = (2*way_point_1 - 0.5*np.array(start) - 0.5*np.array(end)).tolist()
                way_point_2 = (2*way_point_2 - 0.5*np.array(start) - 0.5*np.array(end)).tolist() # uncomment for cantwell"""
                #way_point_1 = [start[0], end[1]]
                #way_point_2 = [end[0], start[1]]
                #way_point_1, way_point_2 = self.get_waypoint(start, end, mid, gt_map, gt_exp_waypoint)
                way_point_1 = door['way_point_1']
                way_point_2 = door['way_point_2']
                p1_same_side = door['p1_same_side']
                dist_1 = self.get_distance(way_point_1, current_location)
                dist_2 = self.get_distance(way_point_2, current_location)
                #print('bot x{} y{}'.format(current_location[0], current_location[1]))
                #print('way1 {} {}'.format(way_point_1, dist_1))  # dist_1
                #print('way2 {} {}'.format(way_point_2, dist_2))  # dist_2
                #if cross_flag_exist and i != cross_node_idx:
                #    self.g.add_edges([[cross_node_idx, i]])
                #else:
                self.g.add_edges([[self.current_node_id, i]])
                edge_idx = self.g.ecount() - 1
                take_way_1 = False
                take_way_2 = False
                print(door)
                if not door['cross_flag']:
                    if p1_same_side:#self.if_same_side(current_location, way_point_1, start, end):#dist_1<dist_2:  # means way point 1 is in the room
                        # self.if_same_side(current_location, way_point_1, start, end):
                        self.g.es[edge_idx]['way_point'] = way_point_2
                        self.g.vs[self.current_node_id]['room_entry'].append(way_point_1)
                        self.g.vs[i]['room_entry'].append(way_point_2)
                        take_way_2 = True
                    else:
                        self.g.es[edge_idx]['way_point'] = way_point_1
                        self.g.vs[self.current_node_id]['room_entry'].append(way_point_2)
                        self.g.vs[i]['room_entry'].append(way_point_1)
                        take_way_1 = True
                else:  # else means the robot has going through the door
                    return_flag = True
                    if not p1_same_side:#self.if_same_side(current_location, way_point_1, start, end):#dist_1<dist_2:  # means way point 1 is in the room
                        # self.if_same_side(current_location, way_point_1, start, end):
                        self.g.es[edge_idx]['way_point'] = way_point_2
                        self.g.vs[self.current_node_id]['room_entry'].append(way_point_1)
                        self.g.vs[i]['room_entry'].append(way_point_2)
                        take_way_2 = True
                    else:
                        self.g.es[edge_idx]['way_point'] = way_point_1
                        self.g.vs[self.current_node_id]['room_entry'].append(way_point_2)
                        self.g.vs[i]['room_entry'].append(way_point_1)
                        take_way_1 = True

                #if cross_flag_exist and i != cross_node_idx:
                #    self.g.add_edges([[i, cross_node_idx]])
                #else:
                self.g.add_edges([[i, self.current_node_id]])
                edge_idx = self.g.ecount() - 1
                if take_way_1:
                    self.g.es[edge_idx]['way_point'] = way_point_2
                if take_way_2:
                    self.g.es[edge_idx]['way_point'] = way_point_1
                current_location_reverse = current_location.copy()
                current_location_reverse.reverse()
                detecting_point = self.g.vs[i]['room_entry'][0].copy()
                detecting_point.reverse()

                #f_list, _, _ = self.frontier_detector.frontier_detection(np.array(detecting_point), np.array(current_location_reverse),
                #                                                        np.rint(gt_map), np.rint(gt_exp), lmb,
                #                                                         door_list)

                #if not return_flag:
                #    if len(f_list) == 0:
                #        self.g.vs[i]['room_status'] = 'explored'

                """if dist_1<dist_2:
                    self.g.es[edge_idx]['way_point'] = way_point_1
                else:
                    self.g.es[edge_idx]['way_point'] = way_point_2"""
            cross_flag_exist = False
        self.current_node_id = tmp_current_node_idx_store
        self.v_num = self.g.vcount()
        #layout = self.g.layout('kk')
        plt.ion()
        fig, ax = plt.subplots()
        ig.plot(self.g, target=ax,
                vertex_label=['{}, {}, {}'.format(i, room_status, self.g.vs[i]['room_entry']) for i, room_status in enumerate(self.g.vs['room_status'])]
                ,edge_label=['{}, {}'.format(way_point,i) for i, way_point in enumerate(self.g.es['way_point'])])
        plt.show()
        plt.pause(2)
        plt.ioff()
    #plt.close()
        entry_dist = 100000
        if len(self.g.vs[self.current_node_id]['room_entry']) != 0:
            for node_entry in self.g.vs[self.current_node_id]['room_entry']:
                room_entry = node_entry.copy()
                room_entry.reverse()
                dist = self.get_distance(room_entry, current_location)
                if dist < entry_dist:
                    room_entry.reverse()
                    min_entry = room_entry
        else:
            current_location.reverse()
            min_entry = current_location
        return min_entry, return_flag#self.g.vs[self.current_node_id]['room_entry'][0]


    def if_same_side(self, current_position, target_position, start, end):
        start = np.flip(np.array(start))
        end = np.flip(np.array(end))
        current_position = np.flip(np.array(current_position))
        target_position = np.flip(np.array(target_position))
        print('current_position {}'.format(current_position))
        print('target_position {}'.format(target_position))
        print('start {}'.format(start))
        print('end {}'.format(end))
        door_dir = end - start
        start_bot = current_position - start
        start_target = target_position - start
        result = np.dot(np.cross(door_dir, start_bot), np.cross(door_dir, start_target))
        print('result {}'.format(result))
        if result > 0:
            return True
        else:
            return False

    def get_waypoint(self, start, end, mid, gt_map, gt_exp):
        # this method is not used in this class anymore, the waypoint will be provided by door detection
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
        waypoint1 = (mid + 0.7*door_length*vertical_dir).astype(int).tolist()
        waypoint2 = (mid - 0.7*door_length*vertical_dir).astype(int).tolist()  # 0.7 is better, but failed in cantwell
        scale_factor = 1
        while gt_exp[waypoint1[1], waypoint1[0]] != 1 or gt_map[waypoint1[1], waypoint1[0]] == 1:  #gt_map[waypoint1[1], waypoint1[0]] == 1 and
            waypoint1 = (mid + 0.9**scale_factor * door_length * vertical_dir).astype(int).tolist()
            scale_factor += 1
        scale_factor = 1
        while gt_exp[waypoint2[1], waypoint2[0]] != 1 or gt_map[waypoint2[1], waypoint2[0]] == 1:  # gt_exp[waypoint1[1], waypoint1[0]] != 1 or gt_map[waypoint1[1], waypoint1[0]] == 1
            waypoint2 = (mid - 0.9**scale_factor * door_length * vertical_dir).astype(int).tolist()
            scale_factor += 1
        return waypoint1, waypoint2

    def delete_vertice(self, v_id):
        self.g.delete_vertices(v_id)
        v_num = self.g.vcount()
        layout = self.g.layout('kk')
        fig, ax = plt.subplots()
        """ig.plot(self.g, layout=layout, target=ax, vertex_label=[i for i in range(v_num)])
        plt.show()"""

    def choose_door(self, current_location):  # current_location is [y,x], global frame unit is pix
        current_location.reverse()
        # first check directly connected vertices
        connected_list = self.g.incident(self.current_node_id, mode='out')
        min_distance = 10000
        goal = []
        no_exp = False
        for edge_idx in connected_list:
            #print(edge_idx)
            way_point = self.g.es[edge_idx]['way_point']
            target_v_id = self.get_connected_v(edge_idx, self.current_node_id)
            if self.g.vs[target_v_id]['room_status'] != 'unexplored':
                continue
            distance = self.get_distance(way_point, current_location)
            #print('distance {}'.format(distance))
            if distance < min_distance:
                goal_edge_idx = edge_idx
                goal.clear()
                goal.append(way_point)
                min_distance = distance
        if min_distance == 10000:  # means no neighbor vertices is unexplored, then we should check the whole topo map
            #print('no surrounding exp')
            goal = []  # means no room to explore
            rest_unexp_list = []#self.g.vs.select(room_status_eq='unexplored')
            for node_idx in range(self.g.vcount()):
                if self.g.vs[node_idx]['room_status'] != 'unexplored':
                    continue
                rest_unexp_list.append(node_idx)
            #print('rest {}'.format(rest_unexp_list))
            min_distance_2 = 1000000
            goal_node_idx = None
            for node_idx in rest_unexp_list:
                entry_list = self.g.vs[node_idx]['room_entry']
                passing_path = self.g.get_shortest_paths(self.current_node_id, to=node_idx, mode='out',
                                                          output='epath')

                distance = 0
                last_pos = current_location
                for e_idx in passing_path[0]:
                    next_pos = self.g.es[e_idx]['way_point']
                    distance += self.get_distance(last_pos, next_pos)
                    last_pos = next_pos
                #print(entry_list)
                #for entry_point in entry_list:
                #    distance = self.get_distance(entry_point, current_location)
                print('dist {}'.format(distance))
                if distance < min_distance_2:
                    min_distance_2 = distance
                    goal_node_idx = node_idx

            """
                        for node_idx in rest_unexp_list:
                entry_list = self.g.vs[node_idx]['room_entry']
                passing_path = self.g.get_shortest_paths(self.current_node_id, to=node_idx, mode='out',
                                                          output='epath')
                
                print('passing path {}'.format(passing_path))
                #print(entry_list)
                for entry_point in entry_list:
                    distance = self.get_distance(entry_point, current_location)
                    print('dist {}'.format(distance))
                    if distance < min_distance_2:
                        min_distance_2 = distance
                        goal_node_idx = node_idx
            """
            if goal_node_idx:
                shortest_path = self.g.get_shortest_paths(self.current_node_id, to=goal_node_idx, mode='out', output='epath')
                #print('goal_idx {}'.format(goal_node_idx))
                #print('spath {}'.format(shortest_path))
                for route in shortest_path[0]:
                    goal.append(self.g.es[route]['way_point'])
                self.g.vs[self.current_node_id]['room_status'] = 'explored'
                self.current_node_id = goal_node_idx
                self.g.vs[self.current_node_id]['room_status'] = 'exploring'
            else:
                no_exp = True
        else:
            # update the graph information
            self.g.vs[self.current_node_id]['room_status'] = 'explored'
            self.current_node_id = self.get_connected_v(goal_edge_idx, self.current_node_id)
            self.g.vs[self.current_node_id]['room_status'] = 'exploring'

        if no_exp:
            goal = []
            self.g.vs[self.current_node_id]['room_status'] = 'explored'

        return goal  # goal is list of [x,y] global frame unit is pix

    def get_connected_v(self, edge_idx, start_node_idx):
        """
        get edg_idx and start_node_idx and return the idx of another node idx
        Args:
            edge_idx:

        Returns:

        """
        edge_list = self.g.get_edgelist()
        out_edge = edge_list[edge_idx]
        out_edge = list(out_edge)
        out_edge.remove(start_node_idx)
        return out_edge[0]

    def get_distance(self, point_1, point_2):
        return ((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2) ** 0.5

    def stop_exp(self):
        stop_flag = True
        for room_status in self.g.vs['room_status']:
            if room_status != 'explored':
                stop_flag = False
                break
        return stop_flag

    def use_exist_topomap(self, topomap):
        self.g = topomap
        self.use_topo = True

if __name__ == '__main__':
    prior_door_list = [{'start': [238, 255], 'end': [228, 244]}, {'start': [252, 296], 'end': [261, 306]},
                       {'start': [266, 307], 'end': [276, 298]}, {'start': [154, 180], 'end': [143, 190]},
                       {'start': [128, 178], 'end': [138, 188]}, {'start': [107, 156], 'end': [95, 146]},
                       {'start': [187, 224], 'end': [205, 216]}, {'start': [172, 236], 'end': [160, 252]}]  # [x, y]
    for content in prior_door_list:
        content['mid'] = ((np.array(content['start']) + np.array(content['end'])) / 2).tolist()
    topo = Topomap_construction()
    topo.add_room(prior_door_list[:1], [240, 240])
    print(topo.choose_door([240,240]))
    topo.add_room([],[24,24])
    print(topo.stop_exp())
    #topo.add_room(([1,1]))
    #topo.add_room(([1, 1, 1]))
