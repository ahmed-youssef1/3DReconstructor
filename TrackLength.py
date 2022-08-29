import numpy as np

# Function that finds length of track through finding closest point on a line to a point#
# this is used to find closest point of (fit line) to (minimum) and (maximum) data point
# based on: https://math.stackexchange.com/questions/13176/how-to-find-a-point-on-a-line-closest-to-another-given-point

# (Example):

# ln_pt1 -> first point on the line
# ln_pt2 -> second point on the line
# min_data_pt -> Minimum data point (along x axis), we find closest point on fit line to that point
# max_data_pt -> Maximum data point (along x axis), we find closest point on fit line to that point

def track_length(ln_pt1, ln_pt2, min_data_pt, max_data_pt):
    ab_vec = ln_pt2 - ln_pt1
    ap1_vec = ln_pt1 - min_data_pt
    ap2_vec = ln_pt1 - max_data_pt

    t1 = (np.dot(ab_vec, ap1_vec)) / (np.dot(ab_vec, ab_vec))  # scalar magnitude, track start
    t2 = (np.dot(ab_vec, ap2_vec)) / (np.dot(ab_vec, ab_vec))  # scalar magnitude, track end

    g1 = ln_pt1 - (t1 * ab_vec)  # closest point to track start
    g2 = ln_pt1 - (t2 * ab_vec)  # closest point to track end

    track_len = np.linalg.norm(g1.astype(float) - g2.astype(float))  # distance between two points (track length)

    return track_len