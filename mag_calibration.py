import pandas as pd
import numpy as np
from data_utils import *
from config_parsers import update_sensor_config


MAG_ID = 3
CALIBRATION_FILE = "Data/Calibration/Mag_Calibration_20201113T1806.txt"
NOISE_FILE = "Data/Calibration/Sensor_Noise_20201113T1842.txt"
SAVE_FILE = "Configs/sensor_config_20201029T1909.xml"
MAG_FIELD = 55.8586  # [uT] Wellington


# Calibration data clip points (ms)
calib_start = 100
calib_end = 3e5

# Noise clip times
noise_start = 1e5
noise_end = 5.5e6


def reconstruct_symmetric_mat(upper_entries):
    upper_entries = upper_entries.flatten()
    Q = np.zeros((3, 3))
    Q[np.triu_indices(3)] = upper_entries[:6]
    Q[1:, 0] = Q[0, 1:]
    Q[2, 1] = Q[1, 2]
    return Q


def levenberg_marquardt(f, df, x, N):
    x = x.reshape(-1, 1)
    max_iters = 100
    tol = 0.01
    count = 0
    lambd = 0

    # Compute initial residuals
    prev_residuals = np.sum(f(x) ** 2) / N
    while count < max_iters:
        count += 1
        F = f(x)
        J = df(x)
        JJ = J.T @ J
        A = (JJ + lambd * np.diag(JJ))
        b = J.T @ F
        try:
            x = x + np.linalg.solve(A, b)
        except Exception as e:
            print("Singular matrix encountered")
            lambd = 1
            continue
        residuals = np.sum(f(x) ** 2) / N
        if abs(residuals - prev_residuals) < tol and lambd < 0.8:
            return x
        prev_residuals = residuals
    print(f"Max iterations exceeded")
    return x


def compute_calibration_params(mag_data):
    data = mag_data.iloc[:, 1:].to_numpy()
    H = np.ones((mag_data.shape[0], 1)) * MAG_FIELD

    def objective(gamma):
        K = reconstruct_symmetric_mat(gamma)
        f2 = (data + gamma[6:, 0]) @ K  # m * 3
        f = np.sqrt(np.sum(f2 * f2, axis=1, keepdims=True))   #mx1
        F = H - f
        return F

    def jacobian(gamma):
        gamma = gamma.flatten()
        J = np.zeros((data.shape[0], 9))
        # Intermediate variables
        A = gamma[0] * (data[:, 0:1] + gamma[6]) + gamma[1] * (data[:, 1:2] + gamma[7]) + gamma[2] * (data[:, 2:3] + gamma[8])
        B = gamma[1] * (data[:, 0:1] + gamma[6]) + gamma[3] * (data[:, 1:2] + gamma[7]) + gamma[4] * (data[:, 2:3] + gamma[8])
        C = gamma[2] * (data[:, 0:1] + gamma[6]) + gamma[4] * (data[:, 1:2] + gamma[7]) + gamma[5] * (data[:, 2:3] + gamma[8])
        K = reconstruct_symmetric_mat(gamma[:6])
        b = np.linalg.norm(K @ (data + gamma[np.newaxis, 6:]).T, axis=0).reshape(-1, 1)
        # Construct Jacobian
        J[:, 0:1] = (data[:, 0:1] + gamma[6]) * (A / b)
        J[:, 3:4] = (data[:, 1:2] + gamma[7]) * (B / b)
        J[:, 5:6] = (data[:, 2:3] + gamma[8]) * (C / b)
        J[:, 1:2] = ((data[:, 1:2] + gamma[7]) * A + (data[:, 0:1] + gamma[6]) * B) / b
        J[:, 2:3] = ((data[:, 2:3] + gamma[8]) * A + (data[:, 0:1] + gamma[6]) * C) / b
        J[:, 4:5] = ((data[:, 2:3] + gamma[8]) * B + (data[:, 1:2] + gamma[7]) * C) / b
        J[:, 6:7] = A / b
        J[:, 7:8] = B / b
        J[:, 8:9] = C / b
        return J

    gamma0 = np.array([1, 0, 0, 1, 0, 1, 0, 0, 0])
    params = levenberg_marquardt(objective, jacobian, gamma0, data.shape[0])
    K = reconstruct_symmetric_mat(params[:6])
    biases = -params[6:]
    return K, biases


def compute_covariance(noise_data):
    """
    Commpute the variance for each axis and form a
    diagonal covariance matrix
    """
    try:
        noise_data = noise_data.to_numpy()
    except AttributeError:
        noise_data = np.array(noise_data)
    variances = np.std(noise_data[:, 1:], axis=0) ** 2
    covariance = np.diag(variances)
    return covariance


def magnetometer_calibration(visualize=True):
    """
    Perform a magnetometer calibration using a single datafile
    which contains data from randomly rotating the device
    """ 
    print("\nRunning Magnetometer Calibration")
    calibration_dict = {}

    # Load and visualize calibration data
    mag_dict = extract_single_imu_sensor_data(CALIBRATION_FILE, MAG_ID, zero_times=True)
    mag_data = pd.DataFrame.from_dict(mag_dict)
    mag_data = get_subsets_between_times(mag_data, [(calib_start, calib_end)])
    time_steps, rates = compute_logging_rates(mag_dict)
    print(f"Log size: {mag_data.shape[0]}")
    print(f"Time step (ms): min:{min(time_steps)}, max:{max(time_steps)}, avg:{np.mean(time_steps):.2f}")
    print(f"Average logging rate (Hz): {1000*np.mean(rates):.2f}")
    if visualize:
        visualize_3axis_timeseries(mag_data, MAG_ID)
        visualize_3D_scatter(mag_data, MAG_ID)
        visualize_2D_projection(mag_data, MAG_ID)

    # Calibration
    scale_inv, biases = compute_calibration_params(mag_data)
    calibration_dict["scale"] = scale_inv
    calibration_dict["biases"] = biases

    # Load and visualize noise data
    noise_data = extract_single_imu_sensor_data(NOISE_FILE, MAG_ID, zero_times=True)
    noise_data = pd.DataFrame.from_dict(noise_data)
    if visualize: visualize_3axis_timeseries(noise_data, MAG_ID)

    # Noise
    noise_data = get_subsets_between_times(noise_data, [(noise_start, noise_end)])
    covariance = compute_covariance(noise_data)
    calibration_dict["covariance"] = covariance

    # Results Visualization
    corrected_data = batch_measurement_correction(mag_data, biases, scale_inv)
    if visualize:
        visualize_3D_scatter(corrected_data, MAG_ID)
        visualize_2D_projection(corrected_data, MAG_ID)
    print(f"Covariance: \n{covariance}")

    # Saving
    save_params = input("Save calibration params?: ")
    if save_params.lower() in ("y","yes"):
        update_sensor_config(SAVE_FILE, MAG_ID, calibration_dict)
        print("Params saved")


def main():
    magnetometer_calibration()



if __name__ == "__main__":
    main()


# def gauss_newton(f, df, x):
#     x = x.reshape(-1, 1)
#     max_iters = 100
#     tol = 0.1
#     count = 0
#     x_prev = x + 10 * tol
#     while count < max_iters:
#         count += 1
#         if np.linalg.norm(x - x_prev) < tol:
#             print(f"Converge Tolerance: {tol} reached")
#             return x
#         x_prev = x
#         F = f(x)
#         DF = df(x)
#         A = (DF.T @ DF)
#         b = DF.T @ (-F)
#         try:
#           x = x + np.linalg.solve(A, b)
#         except Exception as e:
#           print("Singular matrix encountered")
#           x = np.array([])
#           return x
#     final_res = np.linalg.norm(f(x))
#     print(f"Max iterations exceeded, final residual: {final_res:.2f}")
#     return x
#
#
# def compute_calibration_params(mag_data, params_guess):
#     H = np.ones((mag_data.shape[0], 1)) * (MAG_FIELD ** 2)
#     XYZ = mag_data.iloc[:, 1:].to_numpy()
#
#     def objective(beta):
#         ellipse = np.sum(((XYZ - beta[:3].flatten()) / beta[3:].flatten()) ** 2, axis=1, keepdims=True)
#         loss = H - ellipse
#         return loss
#
#     def jacobian(beta):
#         J = np.zeros((XYZ.shape[0], 6))
#         J[:, :3] = 2 * (XYZ - beta[:3].flatten()) / (beta[3:].flatten() ** 2)
#         J[:, 3:] = 2 * ((XYZ - beta[:3].flatten()) ** 2) / (beta[3:].flatten() ** 3)
#         return J
#
#     cal_params = gauss_newton(objective, jacobian, params_guess)
#     P = np.diag(cal_params[3:].flatten())
#     bias = cal_params[:3]
#     return P, bias

# def compute_calibration_params(mag_data):
#     data = mag_data.iloc[:, 1:].to_numpy()
#     Y = np.vstack((data[:,0]**2, 2*data[:,0]*data[:,1], 2*data[:,1]*data[:,2], data[:,1]**2, 2*data[:,1]*data[:,2], data[:,2]**2)).T  # (M x 6)
#     Y = np.concatenate((Y, data, np.ones((data.shape[0], 1))), axis=1)
#     H = np.eye(10)  # norm constraint
#     H[[1, 2, 4], [1, 2, 4]] = np.sqrt(2)
#     H_inv = np.linalg.inv(H)
#     U, S, Vh = np.linalg.svd(Y @ H_inv, full_matrices=False)
#     beta = H_inv @ Vh[-1, :].reshape(-1, 1)  # parameters
#
#     # Reconstruct quadratic solution
#     Q = np.zeros((3, 3))
#     Q[np.triu_indices(3)] = beta[:6, 0]
#     Q[1:, 0] = Q[0, 1:]
#     Q[2, 1] = Q[1, 2]
#     q = beta[6:9]
#     k = beta[-1,0]
#
#     # Specify ellipse params
#     biases = (-1 / 2) *  np.linalg.solve(Q, q)
#     w, V = np.linalg.eig(Q)
#     D = np.diag(w)
#     alpha = 4 * (MAG_FIELD ** 2) / (4 * k - (q.reshape(1, -1) @ V) @ np.linalg.inv(D) @ (V.T @ q))
#     alpha = alpha.item()
#     scale_inv = V @ np.sqrt(alpha * D) @ V.T
#     return scale_inv, biases