import numpy as np
import math

import matplotlib
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use("TkAgg")  # must add to see something
from matplotlib import pyplot as plt

from skimage.measure import LineModelND, ransac
from sklearn.decomposition import PCA

from PolyFit import polyfit_3d
from TrackLength import track_length

# Function for multivariate sequential RANSAC
def ransac_3d(xyz_coordinates, data_3d, plot):
    min_samples_3d = 2
    min_points_cluster_3d = 20  # minimum points to consider it a cluster

    az_angle = []  # list to store azimuthal angles
    polar_angle = []  # list to polar angles
    track_len_ls = []  # list to store tracks lengths
    energy_bin_ls = []  # list to store energy loss values of track
    energy_sum_ls = []  # list to store total energy loss of track
    ransac_inliers = []  # list to store inliers
    ran_outliers = []  # list to store inliers
    inliers_idx_ls = []  # list to store inliers idx
    fit_3d = []

    for i in range(0, len(xyz_coordinates)):

        if len(xyz_coordinates) >= min_samples_3d:
            # robustly fit line only using inlier data with RANSAC algorithm
            model_robust, inliers = ransac(xyz_coordinates, LineModelND, min_samples=min_samples_3d,
                                           residual_threshold=4, max_trials=500)
            outliers = inliers == False

            xyz_coords_inliers = xyz_coordinates[inliers]
            xyz_coords_outliers = xyz_coordinates[outliers]

            if len(xyz_coords_inliers) >= min_points_cluster_3d:

                # Fit line in 3D. Based on:https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_3d.html
                # run PCA to find best fit line
                pca = PCA(n_components=1)
                pca.fit(xyz_coords_inliers)
                eigenvectors = pca.components_  # principle components (unit vectors), where i-th vector is ..
                # the direction of line that best fits the data while being orthogonal to the first i-1 vectors

                # get mean of coordinates
                origin_point = np.mean(xyz_coords_inliers, axis=0)  # origin point = mean
                euclidean_distance = np.linalg.norm(xyz_coords_inliers - origin_point, axis=1)
                max_elu = np.max(euclidean_distance)

                best_fit = np.vstack((origin_point - eigenvectors * max_elu,
                                      origin_point + eigenvectors * max_elu))
                fit_3d.append(best_fit)

                polyfit_3d_call = polyfit_3d(xyz_coords_inliers[:, 0], xyz_coords_inliers[:, 1],
                                             xyz_coords_inliers[:, 2], 1)

                first_fn = polyfit_3d_call[0]
                second_fn = polyfit_3d_call[1]
                x_new = polyfit_3d_call[2]  # this is original x + x_val from the function above

                ###########################
                # ***** Track length *****#
                ###########################

                # Two points on fit line
                line_pt1 = np.array(best_fit[0])
                line_pt2 = np.array(best_fit[1])

                # Find min & max data points
                min_pt_idx = np.where(xyz_coords_inliers[:, 0] == xyz_coords_inliers[:, 0].min())
                max_pt_idx = np.where(xyz_coords_inliers[:, 0] == xyz_coords_inliers[:, 0].max())

                min_pt = xyz_coords_inliers[min_pt_idx][0]
                max_pt = xyz_coords_inliers[max_pt_idx][0]

                # **** Track length ****
                # the function track_length uses Euclidean distance based on: https://en.wikipedia.org/wiki/Euclidean_distance
                track_len = track_length(line_pt1, line_pt2, min_pt, max_pt)
                track_len_ls.append(track_len)

                # **** Beam angle (or polar) ****
                # Angle between track and beam line
                # To find angle, three points are needed: track start and end point, and third point is end point of beam

                track_start_idx = np.where(best_fit[:, 0] == best_fit[:, 0].min())[0][0]  # find index of track start
                track_end_idx = np.where(best_fit[:, 0] == best_fit[:, 0].max())[0][0]  # find index of track end

                O = np.array([[x_new[0]], [first_fn[0]], [second_fn[0]]])

                # Second point (end of the track)
                A = np.array([[x_new[-1]], [first_fn[-1]], [second_fn[-1]]])

                # Third point (end point of orthogonal line) - in the same direction as beam line
                B = np.array([[x_new[0]], [128], [second_fn[0]]])

                # Vectors
                OA_vec = A - O
                OB_vec = B - O

                # vectors magnitudes
                OA_vec_mag = math.sqrt(OA_vec[0] * OA_vec[0] + OA_vec[1] * OA_vec[1] + OA_vec[2] * OA_vec[2])
                OB_vec_mag = math.sqrt(OB_vec[0] * OB_vec[0] + OB_vec[1] * OB_vec[1] + OB_vec[2] * OB_vec[2])

                # Normalisation of vectors (unit vectors)
                OA_unit_vec = (OA_vec[0] / OA_vec_mag, OA_vec[1] / OA_vec_mag, OA_vec[2] / OA_vec_mag)
                OB_unit_vec = (OB_vec[0] / OB_vec_mag, OB_vec[1] / OB_vec_mag, OB_vec[2] / OB_vec_mag)

                # Acute angle
                # dot_pro = np.dot(OA_unit_vec, OB_unit_vec)  # gives same result (takes just a bit longer)
                dot_pro = OA_unit_vec[0] * OB_unit_vec[0] + OA_unit_vec[1] * OB_unit_vec[1] + OA_unit_vec[2] * \
                          OB_unit_vec[2]  # dot product of the two unit vectors
                theta = math.degrees(math.acos(dot_pro))  # angle in degrees
                polar_angle.append(theta)

                # **** Azimuthal angle ****
                # Angle between track and line perpendicular
                # To find angle, three points are needed: track start and end point, and third point is end point in Z direction

                # point 1 and 2 will be like above, but point 3 will change
                # Third point in direction of Z
                C = np.array([[x_new[0]], [first_fn[0]], [500]])

                # Vectors
                OC_vec = C - O

                # vectors magnitudes
                OB_vec_mag = math.sqrt(OC_vec[0] * OC_vec[0] + OC_vec[1] * OC_vec[1] + OC_vec[2] * OC_vec[2])

                # Normalisation of vectors (unit vectors)
                OC_unit_vec = (OC_vec[0] / OB_vec_mag, OC_vec[1] / OB_vec_mag, OC_vec[2] / OB_vec_mag)

                # Acute angle
                dot_pro = OA_unit_vec[0] * OC_unit_vec[0] + OA_unit_vec[1] * OC_unit_vec[1] + OA_unit_vec[2] * \
                          OC_unit_vec[2]  # dot product of the two unit vectors
                theta_azi = math.degrees(math.acos(dot_pro))  # angle in degrees
                az_angle.append(theta_azi)

                # **** Total energy deposited by a track (particle) (E) ****
                # We need to find values of "energy_bins" when values of x,y,z were xyz_coords_inliers
                # That is the total energy deposited of track
                for i, j, n in xyz_coords_inliers:
                    data_3d_idx = np.where((data_3d[:, 0] == i) & (data_3d[:, 1] == j) &
                                           (data_3d[:, 2] == n))
                    energy_bins = data_3d[data_3d_idx][:, 3]
                    energy_bin_ls.append(energy_bins)

                energy_bins_all = np.concatenate(energy_bin_ls)
                energy_bins_all = energy_bins_all
                energy_sum = np.sum(energy_bins_all)
                energy_sum_ls.append(energy_sum)

                inliers_ls = np.column_stack(
                    [xyz_coords_inliers[:, 0], xyz_coords_inliers[:, 1], xyz_coords_inliers[:, 2]])

                ransac_inliers.append(inliers_ls)

            xyz_coordinates = xyz_coordinates[~inliers]  # remaining points

            if len(xyz_coordinates) < min_points_cluster_3d:  # break the loop if not enough points
                break

    # Extract outliers
    if len(ransac_inliers) > 0:

        ransac_inliers = np.array(ransac_inliers)
        ransac_inliers = np.concatenate(ransac_inliers)

        for x, y, z in ransac_inliers:  # find index of "ransac_inliers" in "xyz"
            inliers_idx = np.where((xyz_coordinates[:, 0] == x) * (xyz_coordinates[:, 1] == y) * (xyz_coordinates[:, 2] == z))
            inliers_idx_ls.append(inliers_idx)

        ransac_outliers = np.delete(xyz_coordinates, inliers_idx_ls, 0)  # subtract inliers from data to get outliers
        ran_outliers.append(ransac_outliers)


    else:
        pass

    # Plot
    # Here, we treat only up to 4 cluster, but this can be improved
    if plot:

        cluster_3d = np.array(['cluster-1', 'cluster-2', 'cluster-3', 'cluster-4'])

        colors_3d = "rgbky"
        idx_3d = 0

        label_cluster = cluster_3d[idx_3d]
        color = colors_3d[idx_3d % len(colors_3d)]
        idx_3d += 1

        figure = plt.figure()
        figure.suptitle('Clustering and fitting of tracks')
        ax2 = figure.add_subplot(111, projection='3d')

        ax2.plot(fit_3d[0][:, 1], fit_3d[0][:, 0], fit_3d[0][:, 2], 'r')
        ax2.scatter(ransac_inliers[:, 1], ransac_inliers[:, 0], ransac_inliers[:, 2], color + ".",
                    label=label_cluster)
        ax2.plot(ran_outliers[0][:, 1], ran_outliers[0][:, 0], ran_outliers[0][:, 2], '.k', label='')

        ax2.legend(loc='lower left')

        ax2.set_xlabel('X [mm]', fontsize=10)
        ax2.set_ylabel('Y [mm]', fontsize=10)
        ax2.set_zlabel('Z [mm]', fontsize=10)

        ax2.set_ylim([0, 64])
        ax2.set_xlim([0, 128])
        ax2.set_zlim([0, 600])

        plt.grid()
        plt.savefig('fig1.pdf')
        plt.show()  # for live view

    else:
        pass

    return track_len_ls, polar_angle, energy_sum_ls, az_angle, fit_3d, ransac_inliers, ran_outliers
