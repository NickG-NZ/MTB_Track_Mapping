import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image

from attitude_utils import quat2Euler, quatActiveRot
from coordinate_utils import posGeodetic2Ecef, ecef2Ned


def plot_orientation_timeseries(track_map, raw_map, ax=None):
    """
    Plots the bike attitude as Euler angles for the entire trajectory
    in the track map.
    """
    # Filtered orientation
    rad2deg = 180 / np.pi
    angle_series = [[], [], []]  # psi, theta, phi
    jump_tol = 120  # degrees
    for idx, quat in enumerate(track_map.bike_attitudes):
        psi, theta, phi = quat2Euler(quat)
        angles = [psi, theta, phi]
        for i in range(3):
            a = angles[i] * rad2deg
            if not idx == 0 and abs(a - angle_series[i][-1]) > jump_tol:
                a = 2 * np.pi - a
            angle_series[i].append(a)

    # Raw Gyro integration
    raw_angle_series = [[], [], []]
    for idx, quat in enumerate(raw_map.bike_attitudes):
        psi, theta, phi = quat2Euler(quat)
        angles = [psi, theta, phi]
        for i in range(3):
            a = angles[i] * rad2deg
            if not idx == 0 and abs(a - angle_series[i][-1]) > jump_tol:
                a = 2 * np.pi - a
            raw_angle_series[i].append(a)
    if not ax:
        fig, axis = plt.subplots()
        axis.set_xlabel('time [s]')
        axis.set_title('Bike Orientation')
        axis.grid(True)
    else:
        axis = ax
    times = track_map.map_times
    axis.plot(times, angle_series[0], label='psi')
    axis.plot(times, angle_series[1], label='theta')
    axis.plot(times, angle_series[2], label='phi')
    axis.plot(times, raw_angle_series[0], label='psi_raw')
    axis.plot(times, raw_angle_series[1], label='theta_raw')
    axis.plot(times, raw_angle_series[2], label='phi_raw')
    axis.legend()
    if not ax: return axis


def visualize_path_ecef(track_map, raw_data, ax=None):
    """
    Visualize the 3D path in ECEF coords
    """
    pos_gnss_geo = np.vstack(raw_data['pos'])
    good_rows = [not np.isnan(g) for g in pos_gnss_geo[:, 0]]
    pos_gnss_geo = pos_gnss_geo[good_rows, :]
    pos_gnss_ecef = np.hstack([posGeodetic2Ecef(pos.reshape(-1,1)) for pos in pos_gnss_geo]) - track_map.start_pos_ecef
    pos_gnss_ecef = pos_gnss_ecef.T
    try:
        path = track_map.positions_ecef_array
    except ValueError as e:
        print(e, "Map might be empty")
        path = None
    if not ax:
        fig = plt.figure()
        axis = fig.add_subplot(111, projection='3d')
        axis.grid(True)
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.set_zlabel("z [m]")
    else:
        axis = ax
    axis.plot(path[:,0], path[:,1], path[:,2], label="filtered")
    axis.plot(pos_gnss_ecef[:,0], pos_gnss_ecef[:,1], pos_gnss_ecef[:,2], label="raw")
    axis.legend()
    if not ax: return axis


def image_overlay(track_map, img_path, image_tl_corner, ax=None):
    """
    Overlay the path on a satellite image
    """
    sat_img = np.array(Image.open(img_path))
    path2d = track_map.positions_ned_array[:, 0:2]

    map_root_geo = track_map.start_pos_geodetic
    img_root_geo = np.array([*list(image_tl_corner), map_root_geo[2, 0]]).reshape(-1, 1)
    img_root_ecef = posGeodetic2Ecef(img_root_geo)
    rel_pos_ecef = img_root_ecef - track_map.start_pos_ecef
    rel_pos_ned = ecef2Ned(rel_pos_ecef, map_root_geo)
    y_offset = rel_pos_ned[0, 0] - 10  # N
    x_offset = rel_pos_ned[1, 0] - 18  # E
    size = 762
    extent = [x_offset, x_offset + size, y_offset - size, y_offset]
    if not ax:
        fig, axis = plt.subplots()
        axis.set_xticks([]), axis.set_yticks([])
    else:
        axis = ax
    axis.imshow(sat_img, extent=extent)
    axis.plot(path2d[:, 1], path2d[:, 0], c='b', linewidth=2)  # (plot x: East, y: North)
    if not ax: return axis


def plot_processed_data(processed_data, key, ax=None):
    data = processed_data[key]
    x = []
    y = []
    z = []
    t = []
    for idx, line in enumerate(data):
        if np.isnan(line[0]):
            continue
        t.append(processed_data['time'][idx])
        x.append(line[0])
        y.append(line[1])
        z.append(line[2])
    if not ax:
        fig, axis = plt.subplots()
    else:
        axis = ax
    axis.plot(t, x, label='x')
    axis.plot(t, y, label='y')
    axis.plot(t, z, label='z')
    axis.set_xlabel('time'), axis.set_ylabel(key)
    axis.grid(True)
    axis.legend()
    if not ax: return axis


class MapPathAnimation:

    def __init__(self, track_map, ax=None):
        if ax:
            self.ax = ax
        else:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(projection="3d")
        self.ax.view_init(elev=-150, azim=45)
        self.ax.set_xticks([]), self.ax.set_yticks([]), self.ax.set_zticks([])
        self.track_map = track_map
        self.path = [track_map.positions_ned_array[:, i] for i in range(3)]
        self.ax.set_box_aspect((np.ptp(self.path[0]), np.ptp(self.path[1]), np.ptp(self.path[2])))
        self.bike_axis_lines = self._init_frame()
        self.map_animation = None

    def run_animation(self, save_path=None):
        num_frames = self.path[0].size
        self.map_animation = animation.FuncAnimation(self.fig, self._update_frame, num_frames,
                                                 fargs=(self.track_map, self.bike_axis_lines), interval=10)
        plt.show()
        if save_path:
            mywriter = animation.FFMpegWriter()
            self.map_animation.save(save_path + ".mp4", writer=mywriter)
            print("Animation saving ...")

    @staticmethod
    def _update_frame(idx, track_map, bike_axis_lines):
        position = track_map.positions_ned_array[idx, :]
        q_c2n = track_map.bike_attitudes_array[idx, :].reshape(-1, 1)
        for i in range(3):
            bike_axis = np.tile(position, (2, 1)).T
            unit_vec = np.zeros((3, 1))  # unit vec in bike frame
            unit_vec[i] = 40
            direction = quatActiveRot(q_c2n, unit_vec).flatten()
            bike_axis[:, 1] += direction
            bike_axis_lines[i].set_data(bike_axis[0:2])
            bike_axis_lines[i].set_3d_properties(bike_axis[2])
        return bike_axis_lines

    def _init_frame(self):
        """ Draw first frame of animation """
        # Path line
        self.ax.plot(*self.path, c='k')

        # Fixed Coord axes
        colours = ['r', 'g', 'b']
        for i in range(3):
            axis = np.zeros((3, 2))
            axis[i, 1] = 50
            self.ax.plot(axis[0, :], axis[1, :], axis[2, :], c=colours[i])

        # Bike Coord axes
        position = self.track_map.positions_ned_array[0, :]
        q_c2n = self.track_map.bike_attitudes_array[0, :].reshape(-1, 1)
        bike_axis_lines = []
        for i in range(3):
            bike_axis = np.tile(position, (2, 1)).T
            unit_vec = np.zeros((3, 1))  # unit vec in bike frame
            unit_vec[i] = 40
            direction = quatActiveRot(q_c2n, unit_vec).flatten()
            bike_axis[:, 1] += direction
            bike_axis_lines.append(self.ax.plot(bike_axis[0, :], bike_axis[1, :], bike_axis[2, :], c=colours[i])[0])
        return bike_axis_lines


class AttitudeAnimation:

    def __init__(self, track_map, ax=None):
        if ax:
            self.ax = ax
        else:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(projection="3d")
        self.ax.view_init(elev=-150, azim=45)
        self.ax.set_xticks([]), self.ax.set_yticks([]), self.ax.set_zticks([])
        self.ax.set_box_aspect((1, 1, 0.7))
        self.track_map = track_map
        self.bike_yaxis_line = self._init_frame()
        self.map_animation = None

    def run_animation(self, save_path=None):
        num_frames = self.track_map.bike_attitudes_array.shape[0]
        self.map_animation = animation.FuncAnimation(self.fig, self._update_frame, num_frames,
                                                 fargs=(self.track_map, self.bike_yaxis_line), interval=10)#, blit=True)
        plt.show()
        if save_path:
            mywriter = animation.FFMpegWriter()
            self.map_animation.save(save_path + ".mp4", writer=mywriter)

    @staticmethod
    def _update_frame(idx, track_map, bike_yaxis_line):
        q_c2n = track_map.bike_attitudes_array[idx, :].reshape(-1, 1)
        new_yaxis = np.zeros((3, 2))
        new_yaxis[:, 1] = quatActiveRot(q_c2n, np.array([0, 1, 0]).reshape(-1, 1)).flatten()
        bike_yaxis_line.set_data(new_yaxis[0:2])   # have to set (x,y) separately to z
        bike_yaxis_line.set_3d_properties(new_yaxis[2])
        return bike_yaxis_line

    def _init_frame(self):
        """ Draw first frame of animation """
        # Coord axes
        colours = ['r', 'g', 'b']
        for i in range(3):
            axis = np.zeros((2, 3))
            axis[1, i] = 1
            self.ax.plot(axis[:, 0], axis[:, 1], axis[:, 2], c=colours[i])

        # initial attitude
        q_c2n = self.track_map.bike_attitudes_array[0, :].reshape(-1, 1)
        yaxis = np.zeros((3, 2))
        yaxis[:, 1] = quatActiveRot(q_c2n, np.array([0, 1, 0]).reshape(-1, 1)).flatten()
        bikes_yaxis_line = self.ax.plot(yaxis[0, :], yaxis[1, :], yaxis[2, :], linewidth=2, c="m")[0]
        return bikes_yaxis_line

