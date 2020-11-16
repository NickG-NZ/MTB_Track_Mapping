import os
from mapping import TrackMapper, TrackDataParser, TrackMap
from visualization import *
import data_utils as dutils

# Files
map_cfg_file = os.path.join(os.getcwd(), 'Configs', 'mapping_config.xml')
sensor_cfg_file = os.path.join(os.getcwd(), 'Configs', 'sensor_config_20201029T1909.xml')
# sensor_cfg_file = os.path.join(os.getcwd(), 'Configs', 'sensor_config_default.xml')
# imu_data = os.path.join(os.getcwd(), 'Data/RAW_IMU_Log', 'test_20201104T152431_imu.txt')
# gnss_data = os.path.join(os.getcwd(), 'Data/RAW_GNSS_Log', 'test_20201104T152440_gnss.txt')
imu_data = os.path.join(os.getcwd(), 'Data/RAW_IMU_Log', 'test_20201104T191036_imu.txt')
gnss_data = os.path.join(os.getcwd(), 'Data/RAW_GNSS_Log', 'test_20201104T191042_gnss.txt')
processed_file = os.path.join(os.getcwd(), 'Data/Processed', '20201104T1910.csv')
satellite_img_file = os.path.join(os.getcwd(), "Data/Media", "track_image.png")
video_save_file = os.path.join(os.getcwd(), "Docs/Results", "path_video2")


def main(parse_data=False):
    """
    Run the track mapping application
    """
    if parse_data:
        data_parser = TrackDataParser(map_cfg_file, sensor_cfg_file)
        data_parser.parse_raw_data(imu_data, gnss_data)
        data_parser.save_to_file(processed_file)

    processed_data = dutils.load_processed_data_file(processed_file)

    # Data visualization
    # accel_data = dutils.extract_single_imu_sensor_data(imu_data, 1, zero_times=True, time_in_seconds=True)
    # gyro_data = dutils.extract_single_imu_sensor_data(imu_data, 2, zero_times=True, time_in_seconds=True)
    # mag_data = dutils.extract_single_imu_sensor_data(imu_data, 3, zero_times=True, time_in_seconds=True)
    # dutils.visualize_3axis_timeseries(accel_data, 1)
    # dutils.visualize_3axis_timeseries(gyro_data, 2)
    # dutils.visualize_3axis_timeseries(mag_data, 3)
    # plot_processed_data(processed_data, 'vel')
    # plt.show()
    # plot_processed_data(processed_data, 'pos')
    # plt.show()

    track_map = TrackMap()
    mapper = TrackMapper(track_map, processed_data, map_cfg_file, sensor_cfg_file)
    mapper.run_mapping()
    no_meas_map = TrackMap()
    no_meas_mapper = TrackMapper(no_meas_map, processed_data, map_cfg_file, sensor_cfg_file)
    no_meas_mapper.run_mapping(no_updates=True)

    # Results visualization
    plot_orientation_timeseries(track_map, no_meas_map)
    # img_tl_corner = mapper.map_config["sat_img_tl_corner"]
    # image_overlay(track_map, satellite_img_file, img_tl_corner)
    plt.show()
    # attitude_video = AttitudeAnimation(track_map)
    # attitude_video.run_animation()
    # map_video = MapPathAnimation(track_map)
    # map_video.run_animation()


if __name__ == "__main__":
    main()
