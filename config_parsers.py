"""
Tools for loading and saving configuration files

Currently Supported
====================
* Sensor calibrations (xml)
* Mapping application configuration (xml)
"""

import numpy as np
import xml.etree.ElementTree as ET
from coordinate_utils import deg2rad


def string2numpy(string, dims=None):
    """
    Converts a string of numbers ( eg. "6.7, 1.3, 5.5" )
    into a numpy array with dimensions 'dims'
    """
    array = np.array([float(i) for i in string.split(',')])
    if not dims:
        dim = int(np.sqrt(array.size))
        dims = (dim, dim)
    array = array.reshape(dims)
    return array


def numpy2string(array):
    """
    converts a numpy array to a flattened string
    """
    string = str(list(array.flatten()))[1:-1].replace(' ', '')
    return string


def update_sensor_config(file_path, sensor_id, calibration_values):
    """
    Logs calibration params in an xml calibration file
    Inputs:
    - calibration_values: a dictionary containing the calibration values to save
    """
    try:
        tree = ET.parse(file_path)
    except Exception as e:
        print(f" {e} Failed to open file")
        return None

    sensor_found = False
    root = tree.getroot()
    for sensor in root:
        if sensor.get('id') == str(sensor_id):
            sensor_found = True
            status = sensor.find('status')
            status.set('calibrated', 'true')
            for key, val in calibration_values.items():
                sensor.find(key).text = numpy2string(val)
    if not sensor_found:
        print(f"Failed to find sensor: {sensor_id} in calibration file")
        return 0
    else:
        tree.write(file_path)
    return 1


def load_single_sensor_config(file_path, sensor_id):
    """
    Loads sensor calibration params for a single sensor from calibration xml file
    Returns the params as a dictionary
    """
    calibration_params = {}
    try:
        tree = ET.parse(file_path)
    except Exception as e:
        print(f"{e}: Failed to open calibration file")
        return None

    sensor_found = False
    root = tree.getroot()
    cfg_date = root.get('date')
    for sensor in root:
        if sensor.get('id') == str(sensor_id):
            calibration_params['name'] = sensor.get('name')
            status = sensor.find('status')
            if status.get('calibrated') != "true":
                raise ValueError(f"File does not contain calbration data for sensor: {sensor_id}")
            for param in sensor:
                if param.tag == "status":
                    continue
                elif not param.text:
                    calibration_params[param.tag] = None
                elif param.tag == "biases":
                    calibration_params[param.tag] = string2numpy(param.text, (3, 1))
                elif param.tag in ['covariance', 'bias_covariance', 'scale']:
                    calibration_params[param.tag] = string2numpy(param.text)
                else:
                    raise ValueError(f"Unepxected parameter: {param.tag}")
            sensor_found = True
    if not sensor_found:
        print(f"Failed to find sensor: {sensor_id} in file")
        return None
    print(f"Loaded calibration params<{sensor_id}>: [{cfg_date}]")
    return calibration_params


def load_sensor_config(file, num_sensors=4):
    """
    Loads all sensor params into a dictionary where keys are specified names
    corresponding to the sensor ID numbers.
    Inputs:
     - file, (string) file path
     - sensor_names, (dict) sensor_id: name_string
    """
    sensor_config = {}
    for id_num in range(1, num_sensors + 1):
        sensor_config[id_num] = load_single_sensor_config(file, id_num)
    return sensor_config


def load_mapping_config(file_path):
    """
    Loads mapping configuration params for config xml file
    Returns the params as a dictionary
    """
    map_config = {}
    try:
        tree = ET.parse(file_path)
    except Exception as e:
        print("Failed to open calibration file")
        return None

    root = tree.getroot()
    date = root.get('date')
    track_name = root.get('track_name')
    for group in root:
        if group.tag == "timing":
            map_config["predict_rate"] = float(group.find("predict_rate").text)

        if group.tag == "magnetic_field":
            map_config["mag_field_strength"] = float(group.find("strength").text)
            map_config["mag_field_decl"] = deg2rad(float(group.find("declination").text))
            map_config["mag_field_incl"] = deg2rad(float(group.find("inclination").text))

        if group.tag == "gravity":
            map_config["gravity"] = float(group.find("strength").text)

        if group.tag == "outlier_detection":
            map_config["max_angular_rate"] = float(group.find("max_angular_rate").text)
            map_config["max_specific_force"] = float(group.find("max_specific_force").text)
            map_config["max_mag_field"] = float(group.find("max_magnetic_field").text)

        if group.tag == "sensor_mount_location":
            map_config["sensor_orientation"] = string2numpy(group.find("orientation").text, (3,1))
            map_config["sensor_mount_position"] = string2numpy(group.find("position").text, (3,1))

        if group.tag == "satellite_image":
            map_config["sat_img_tl_corner"] = string2numpy(group.find("top_left_corner").text, (2,))
    print(f"Loaded <mapping configuration params>: [Track: {track_name}, {date}]")
    return map_config
    