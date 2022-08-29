#####################################
#
# Main program for 3DReconstructor
#
#
# Ahmed Youssef
# 29-August-2022
#
#####################################
import time

start_time = time.time()

from PlotHistogram import plot_histo
from PolyFit import polyfit_3d
from RANSAC import ransac_3d
from TrackLength import track_length

import ROOT as R

import operator
import math
import datetime
import numpy as np
import pandas as pd

from itertools import chain

from skimage.measure import LineModelND, ransac
import os
from skimage import io

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import paired_distances

from concurrent.futures import ProcessPoolExecutor
from mpire import WorkerPool

import matplotlib
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use("TkAgg")  # must add to see something
from matplotlib import pyplot as plt

# Read then store pad plane data in RDataFrame
data_df = R.RDataFrame("PadPlaneTree",
                       "/mnt/ksf2/H1/user/u0132845/linux/MVisu_IS581/MVisu_IS581/ROOT_FILES/PadPlaneEventsGETrun_360.root")  # dataframe

# Specify events
data_df = data_df.Filter('EventNumber == 2')
#data_df = data_df.Filter('EventNumber >= 1')  # all events
data_np = data_df.AsNumpy(
    columns=["EventPadXCoord", "EventPadYCoord", "EventNumber", "EventTimestamp", "EventBinValue"])

# Put data in pandas dataframe
data_pd = pd.DataFrame(data_np)

# Filter out (BinValue > 500) -> bad events
bad_events = data_pd.loc[(data_pd.EventBinValue > 500), ["EventNumber"]]
bad_events_dup = bad_events.drop_duplicates()  # dropping duplicate values
bad_events_dup = np.array(bad_events_dup).flatten()

good_events = data_pd.loc[~data_pd.EventNumber.isin(bad_events_dup)]  # good events
good_events_gp = good_events.groupby('EventNumber')  # group good events
group_keys = good_events_gp.groups.keys()

print("Nr of processed events =", good_events_gp.ngroups)
print("These events are: ", good_events_gp.size())

# Lists to store all stats
track_len_all = []
theta_all = []
energy_sum_all = []
energy_sum_len_all = []
event_all = []
azimuthal_angle = []


# Now iterate over groups (good events)
def alpha_stats(group_keys):
    event = good_events_gp.get_group(group_keys)
    X = event['EventPadXCoord']
    Y = event['EventPadYCoord']
    EventNumber = event['EventNumber']
    TimeBin = event['EventTimestamp']
    EnergyBin = event['EventBinValue']  # not yet calibrated

    # Store data & coordinates for 3D RANSAC
    Z = TimeBin
    xyz = np.column_stack([X, Y, Z])

    # store into numpy arrays
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    energy_bin = np.array(EnergyBin)

    data_3d = np.column_stack([X, Y, Z, energy_bin])  # 3D coordinates + energy bins (to be used for total dE)

    # Condition for RANSAC:
    # Idea: if data size > 20 -> run RANSAC & get stats, else: do nothing
    # **Good event**: when 20 - 200 pads fire
    if 20 < xyz[:, 0].size < 200:
        # ransac_3d(xyz, data_3d)

        # Track information
        ransac_call = ransac_3d(xyz, data_3d, True)  # plotting True or False? (use False if a lot of events to avoid memory problems)
        tracks_length = ransac_call[0]
        theta_angle = ransac_call[1]
        energies_sum = ransac_call[2]
        azim_angle = ransac_call[3]

        event_ls = []

        for i in theta_angle:
            event_ls.append(group_keys)

        return tracks_length, theta_angle, energies_sum, event_ls, azim_angle

    else:  # if it's out of (20 - 200) range, it's most likely a bad event, hence passed
        print("Note: This is likely to be a bad event, RANSAC aborted!")
        pass


with WorkerPool(n_jobs=None) as pool:
    alpha_call = [x for x in pool.map(alpha_stats, group_keys, progress_bar=True) if x is not None]
    alpha_call = np.array(alpha_call)

    track_len = alpha_call[:, 0]
    theta = alpha_call[:, 1]
    energy_sum = alpha_call[:, 2]
    group_keys = alpha_call[:, 3]
    azimu_angle = alpha_call[:, 4]

    track_len = track_len.tolist()
    theta = theta.tolist()
    energy_sum = energy_sum.tolist()
    group_keys = group_keys.tolist()
    azimu_angle = azimu_angle.tolist()

    track_len_all.append(track_len)
    theta_all.append(theta)
    energy_sum_all.append(energy_sum)
    event_all.append(group_keys)
    azimuthal_angle.append(azimu_angle)

# Convert list of lists into just one list with all values
track_len_all = list(chain.from_iterable(track_len_all[0]))
theta_all = list(chain.from_iterable(theta_all[0]))
energy_sum_all = list(chain.from_iterable(energy_sum_all[0]))
event_all = list(chain.from_iterable(event_all[0]))
azimuthal_angle = list(chain.from_iterable(azimuthal_angle[0]))

# Convert to numpy + flatten (reduce to zero dimension)
track_len_all = np.array(track_len_all).flatten()
theta_all = np.array(theta_all).flatten()
energy_sum_all = np.array(energy_sum_all).flatten()
event_all = np.array(event_all).flatten()
azimuthal_angle = np.array(azimuthal_angle).flatten()

# Total energy loss (deposited) per length of track (ΔE/Δx)
energy_sum_len = energy_sum_all / track_len_all
energy_sum_len = energy_sum_len

# plot histogram of angular distribution
# NOTE: you can also create your own histograms from the stored values of energies, lengths, angles, etc.
plot_histo(theta_all)

# how long it took
print("It took", "--- %s seconds ---" % (time.time() - start_time), "to run this code")