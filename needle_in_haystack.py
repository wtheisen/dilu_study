import json
import os
import math
import cv2
import progressbar
import collections

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

from joblib import Parallel, delayed
from visualize_nhs import visualize_nhs, visualizeVotes_toRank

class haystack():
    visualize = False
    visFolder = '.'
    ID_To_Image = None
    partition_dictionary = None
    number_of_results_to_retrieve = 500

    def __init__(self, I, D, image_level_IDs, imageSizes, meta_db, meta_query, visualize=False, visFolder='.', IDToImage=None,
            partitionDictionary=None,numberOfResultsToRetrieve=500):
        pass

    def needle(self, I, D, image_level_IDs, imageSizes, meta_db, meta_query, visualize=False, visFolder='.', IDToImage=None,
        partitionDictionary=None,numberOfResultsToRetrieve=500 ):

        qsize = self.getQuantizationSizes(imageSizes, divis=4, power=.8)
        qsize_matrix = qsize.reshape((image_level_IDs.shape[0], -1, 1))

        invD = self.Dscore(D)

        voteCoords, IDs_contig, centroidMatrix = self.projectVotes(I, invD, image_level_IDs, meta_db, meta_query, imageSizes)

        points_of_clusters, size_of_clusters = self.runDensityClustering(voteCoords, qsize_matrix, image_level_IDs.flatten(),
                                                                    IDs_contig, I.flatten(), removeMultiMatch=True,
                                                                    removeMultiDatabase=True)

        voteScores, unique_vq_bin_ids, clusterSizes, clusters, pointDists, votesums, meandists = NHScore_Vectorized(
            voteCoords, points_of_clusters, invD, meta_db, meta_query, imageSizes, useCenterDists=True,
            usePointCoherence=False, useAngleCoherence=False, query_centroid_matrix=centroidMatrix, visualize=visualize,
            visOutDir=visFolder, visRank=200, IDToImage=IDToImage, d=partitionDictionary)

        sortinds = voteScores.argsort()[::-1]
        sortedScores = voteScores[sortinds]
        sortedIDs = unique_vq_bin_ids[sortinds]

        return sortedIDs, sortedScores, sortedScores.max()

    def performVQ(voteCoordstmp, qsize_matrix, shift, image_level_IDs, IDs_contig, featIDs, removeMultiMatch=False,
                  removeMultiDatabase=False):
        voteCoords = voteCoordstmp.copy()
        if shift == 'right':
            voteCoords = voteCoords.copy()
            voteCoords[:, :, 0] = (voteCoords[:, :, 0] - qsize_matrix[:, :, 0] / 2)
        if shift == 'down':
            voteCoords = voteCoords.copy()
            voteCoords[:, :, 1] = (voteCoords[:, :, 1] - qsize_matrix[:, :, 0] / 2)
        if shift == 'rightdown':
            voteCoords = voteCoords.copy()
            voteCoords[:, :, 0] = (voteCoords[:, :, 0] - qsize_matrix[:, :, 0] / 2)
            voteCoords[:, :, 1] = (voteCoords[:, :, 1] - qsize_matrix[:, :, 0] / 2)
        vq = (voteCoords / qsize_matrix).astype(np.int)
        vq[:, :, 0][voteCoords[:, :,
                    0] < 0] -= 1  # To prevent all positive and negative points near zero bin from quantizing to the same
        vq[:, :, 1][voteCoords[:, :, 1] < 0] -= 1
        vq_arr = np.dstack((vq, IDs_contig)).reshape((-1, 3))
        minBins = vq_arr.min(axis=0)
        vq_arr = (vq_arr - minBins)
        maxBins = vq_arr.max(axis=0)
        vq_bin_ids = np.ravel_multi_index(vq_arr.T, (maxBins[0] + 1, maxBins[1] + 1, IDs_contig.max() + 1))
        IDs_contig_flat = IDs_contig.flatten()
        IDs_flat = image_level_IDs.flatten()
        maxbin = vq_bin_ids.max()
        otm_ids_unique_inds = np.arange(0, len(image_level_IDs))
        if removeMultiMatch:
            # remove items from bins that are all matching to the same query feature
            queryIDsMat = np.arange(0, qsize_matrix.shape[0]).reshape((-1, 1)).repeat(qsize_matrix.shape[1], axis=1)
            # 1-to-many ID
            queryIDs_and_binIDs = np.vstack((queryIDsMat.flatten(), vq_bin_ids)).astype(
                int)  # if features in a cluster match to the same query feature, that is 1-to-many matching, and we want to filter them out
            otm_ids = np.ravel_multi_index(queryIDs_and_binIDs, (qsize_matrix.shape[0], int(maxbin) + 1))
            # we know that the first occurence of a bin will have the highest match value, and that is the one we want to keep
            otm_ids_unique, otm_ids_unique_inds = np.unique(otm_ids, return_index=True)
            # otm_ids_unique_inds is the indexes of all of the feature matches we want to keep!
            vq_bin_ids = vq_bin_ids[otm_ids_unique_inds]
            vq_arr = vq_arr[otm_ids_unique_inds]
            IDs_contig_flat = IDs_contig_flat[otm_ids_unique_inds]
            IDs_flat = IDs_flat[otm_ids_unique_inds]
            featIDs = featIDs[otm_ids_unique_inds]

        if removeMultiDatabase:
            # remove items from bins that are all matching to the same query feature
            # 1-to-many ID
            uniqueFeatIDs, uniqueFeatIDs_inv = np.unique(featIDs, return_inverse=True)
            featIDs_contig = np.arange(0, len(uniqueFeatIDs))[uniqueFeatIDs_inv]
            queryIDs_and_binIDs = np.vstack((featIDs_contig, vq_bin_ids)).astype(
                int)  # if features in a cluster match to the same query feature, that is 1-to-many matching, and we want to filter them out
            otm_ids = np.ravel_multi_index(queryIDs_and_binIDs, (featIDs_contig.max() + 1, int(maxbin) + 1))
            # we know that the first occurence of a bin will have the highest match value, and that is the one we want to keep
            otm_ids_unique, otm_ids_unique_inds2 = np.unique(otm_ids, return_index=True)
            # otm_ids_unique_inds is the indexes of all of the feature matches we want to keep!
            vq_bin_ids = vq_bin_ids[otm_ids_unique_inds2]
            vq_arr = vq_arr[otm_ids_unique_inds2]
            IDs_contig_flat = IDs_contig_flat[otm_ids_unique_inds2]
            IDs_flat = IDs_flat[otm_ids_unique_inds2]
            otm_ids_unique_inds = otm_ids_unique_inds[otm_ids_unique_inds2]

        uniqueBins, bin_inds, bin_inv, bin_counts = np.unique(vq_bin_ids, return_index=True, return_counts=True,
                                                              return_inverse=True)

        counts_thres_inds = bin_counts > 0
        inv_thres_inds = bin_counts[bin_inv] > 0

        # remove items from bins that are all matching to the same query feature

        ids_for_bins = IDs_contig_flat.flatten()[bin_inds][counts_thres_inds]
        bin_counts_thresh = bin_counts[counts_thres_inds]
        uniqueBins_thresh = uniqueBins[counts_thres_inds]
        count_max = bin_counts_thresh.max()
        counts_and_ids = np.vstack((bin_counts_thresh, ids_for_bins))
        sorted_lex = np.lexsort(counts_and_ids)[::-1]
        counts_and_ids = counts_and_ids.T[sorted_lex]
        uniqueImageswithClusters, image_start_inds = np.unique(counts_and_ids[:, 1], return_index=True)
        clusterSortInds = image_start_inds[::-1]
        vq_ids_inv_sortedinds = np.argsort(bin_inv)
        imageids_forsortedinds = IDs_flat[vq_ids_inv_sortedinds]
        vqinv_and_imids = np.vstack((otm_ids_unique_inds[vq_ids_inv_sortedinds], imageids_forsortedinds)).T

        vq_ids_inv_sortedinds = vq_bin_ids[vq_ids_inv_sortedinds]

        idSplitList_raw = np.cumsum(bin_counts[:-1])
        points_of_clusters_raw = np.asarray(np.split(vqinv_and_imids, idSplitList_raw))  # Produces a jagged array of indexes, in the order of uniqueBins_thresh
        reorder_inds = np.arange(len(bin_counts))[counts_thres_inds][sorted_lex][clusterSortInds]
        points_of_clusters = points_of_clusters_raw[reorder_inds]

        return (points_of_clusters, bin_counts[reorder_inds])


    def runDensityClustering(voteCoords, qsize_matrix, image_level_IDs, IDs_contig, featIDs, removeMultiMatch=False,
                             removeMultiDatabase=False):
        points_of_clusters = None
        size_of_clusters = None
        imids_for_clusters = None
        points_of_clusters = None
        size_of_clusters = None
        shiftList = ['None', 'right', 'down', 'rightdown']
        allPointsShiftedClusters = Parallel(n_jobs=len(shiftList))(delayed(self.performVQ)(voteCoords, qsize_matrix, offset, image_level_IDs, IDs_contig, featIDs,removeMultiMatch=removeMultiMatch, removeMultiDatabase=removeMultiDatabase) for offset in shiftList)
        for points_of_clusterst, size_of_clusterst in allPointsShiftedClusters:
            if points_of_clusters is None:
                points_of_clusters = points_of_clusterst.copy()
                size_of_clusters = size_of_clusterst.copy()
            else:
                betterSizes = (size_of_clusterst - size_of_clusters) > 0
                size_of_clusters[betterSizes] = size_of_clusterst[betterSizes]
                points_of_clusters[betterSizes] = points_of_clusterst[betterSizes]

        return points_of_clusters, size_of_clusters

    # points_of_clusters,size_of_clusters = runDensityClustering(voteCoords_final,qsize_matrix,IDs_contig)
    def getQuantizationSizes(imageSizes, divis=6, power=.9):
        qsize = np.power(np.amax(imageSizes, axis=1) / divis, power)
        qsize = np.minimum(np.maximum(qsize.astype(np.int), 5), 250)

        return qsize  # ,qsize_matrix

    def Dscore(D):
        D2 = 1 / (1 + np.sqrt(D))

        D2[np.isinf(D)] = np.nan
        D2[D2 > 100] = np.nan
        return D2

    def projectVotes(I, invD, image_level_IDs, allMetaDatabase, allMetaQuery, imageSizes):
        rotationAngles = (-allMetaQuery[:, :, 3] + allMetaDatabase[:, :, 3]) * math.pi / 180
        scaleFactors = ((allMetaDatabase[:, :, 2] / allMetaQuery[:, :, 2])).reshape((allMetaQuery.shape[0], -1, 1))


        # Calculate centroids for each image ID
        uniqueImageIDs, uniqueImageIDs_inds, uniqueImageIDs_inv, uniqueImageIDs_counts = np.unique(
            image_level_IDs.flatten(), return_index=True, return_inverse=True, return_counts=True)
        imageIDtoContiguous = np.arange(len(uniqueImageIDs))[uniqueImageIDs_inv]  # Map all of the image IDs to a contiguous set, easier for bin counting
        flatIDs = image_level_IDs.flatten()
        flatIDs_contig = imageIDtoContiguous
        IDs_contig = flatIDs_contig.reshape(I.shape)
        goodInds = flatIDs >= 0
        # get back to original image IDs by uniqueImageIDs[contigID]

        image_centroid_weights = np.bincount(flatIDs_contig, weights=invD.flatten())
        # find centroids for each image ID, put in array ordered by uniqueImageIDs
        all_centroids_x = np.bincount(flatIDs_contig,
                                      weights=(allMetaQuery[:, :, 0] * invD).flatten()) / image_centroid_weights
        all_centroids_y = np.bincount(flatIDs_contig,
                                      weights=(allMetaQuery[:, :, 1] * invD).flatten()) / image_centroid_weights
        all_centroids = np.vstack((all_centroids_x, all_centroids_y)).T  # Should by a (#of unique images * 2) sized matrix
        centroid_application_matrix = all_centroids[uniqueImageIDs_inv].reshape((image_level_IDs.shape[0],
                                                                                 image_level_IDs.shape[1],
                                                                                 2))  # Should be a tensor of original match matrix size, with x and y channel

        # Calculate vote coordinates in hough space based on metadata from matched keypoints
        query_rotations1 = np.concatenate((np.cos(rotationAngles).reshape((rotationAngles.shape[0], -1, 1)),
                                           -np.sin(rotationAngles).reshape((rotationAngles.shape[0], -1, 1))), axis=2)
        query_rotations2 = np.flip(query_rotations1 * np.asarray([[[1, -1]]]), axis=2)
        adjusted_queryPoints = (centroid_application_matrix - allMetaQuery[:, :, 0:2]) * scaleFactors  # has x and y channel
        initial_vectors = np.concatenate((
                                         (adjusted_queryPoints * query_rotations1).sum(axis=2).reshape((I.shape[0], -1, 1)),
                                         (adjusted_queryPoints * query_rotations2).sum(axis=2).reshape(I.shape[0], -1, 1)),
                                         axis=2)
        voteCoords_final = allMetaDatabase[:, :, 0:2] + initial_vectors

        qsize = self.getQuantizationSizes(imageSizes)
        qsize_matrix = qsize.reshape((image_level_IDs.shape[0], -1, 1))

        return voteCoords_final, IDs_contig, centroid_application_matrix

    def NHScore_Vectorized(voteCoords, points_of_clusters, Dinv, allMetaDatabase, allMetaQuery, imageSizes,
                           useCenterDists=True, usePointCoherence=True, useAngleCoherence=True, query_centroid_matrix=None,
                           returnClusters=True, visualize=False, IDToImage=None, d=None, visRank=100, visOutDir='.'):
        clusters_concat = np.concatenate(points_of_clusters).astype(int)
        all_vq_bin_ids = clusters_concat[:, 0].flatten()
        imagesForClusters = clusters_concat[:, 1].flatten()
        qsize = getQuantizationSizes(imageSizes)
        flat_voteCoords = voteCoords.reshape((-1, 2))

        if query_centroid_matrix is not None:
            mappedCentroids = query_centroid_matrix.reshape((-1, 2))[all_vq_bin_ids]
        mappedVotes = flat_voteCoords[all_vq_bin_ids]
        mappedMetaDB = allMetaDatabase.reshape((-1, allMetaDatabase.shape[2]))[all_vq_bin_ids]
        mappedMetaQ = allMetaQuery.reshape((-1, allMetaQuery.shape[2]))[all_vq_bin_ids]
        mappedMatchScores = Dinv.flatten()[all_vq_bin_ids]
        mappedVoteScores = mappedMatchScores
        mappedQSizes = qsize[all_vq_bin_ids]

        # after applying the imageID to each votecoord, we find the unique ordering (for bincount) and the unweighted counts of each cluster
        unique_vq_bin_ids, unique_vq_bin_inds, unique_vq_bin_ids_inv, unique_bins_count = np.unique(imagesForClusters,
                                                                                                    return_inverse=True,
                                                                                                    return_index=True,
                                                                                                    return_counts=True)  # Cluster_centers_X[unique_vq_bin_ids_inv] will map cluster centers out to all points

        imagesForClusters_contig = np.arange(0, len(unique_vq_bin_ids))[unique_vq_bin_ids_inv]
        final_image_IDs = unique_vq_bin_ids
        # Cluster Center Vote
        if useCenterDists or usePointCoherence:
            Cluster_centers_X = np.bincount(imagesForClusters_contig, weights=mappedVotes[:, 0]) / unique_bins_count
            Cluster_centers_Y = np.bincount(imagesForClusters_contig, weights=mappedVotes[:, 1]) / unique_bins_count
            Cluster_centers_arr = np.vstack((Cluster_centers_X, Cluster_centers_Y)).T[unique_vq_bin_ids_inv]
        if useCenterDists:
            mappedPointDifs = mappedVotes - Cluster_centers_arr
            mappedPointDists = np.linalg.norm(mappedPointDifs, axis=1)
            mappedProbs = st.norm.pdf((mappedPointDists) / mappedQSizes, scale=1.5) / st.norm.pdf(0, scale=1.5)
            mappedVoteScores = mappedMatchScores * mappedProbs
        meandists = np.bincount(imagesForClusters_contig, weights=mappedPointDists) / unique_bins_count
        voteSums = np.bincount(imagesForClusters_contig, weights=mappedVoteScores)
        voteSums[voteSums == 0] = .00001
        voteScores = voteSums * np.log2(unique_bins_count)  # unique_bins_count*np.log2(voteSums)#

        # Angle Coherence Score
        if useAngleCoherence:
            mappedAngleDifferences = mappedMetaDB[:, 3] - mappedMetaQ[:, 3]
            flipinds = mappedAngleDifferences < 0
            mappedAngleDifferences[flipinds] = (360 + mappedAngleDifferences[flipinds])
            angleMeans = (np.bincount(imagesForClusters_contig, weights=mappedAngleDifferences) / unique_bins_count)[
                unique_vq_bin_ids_inv]
            angleSTDs = np.sqrt(np.bincount(imagesForClusters_contig, weights=np.square(
                mappedAngleDifferences - angleMeans)) / unique_bins_count) / unique_bins_count
            angleScore = 1 / (1 + angleSTDs)  # st.norm.pdf(angleSTDs,scale=2.5)/st.norm.pdf(0,scale=2.5)#
            voteScores *= angleScore

        # Point STD score
        if usePointCoherence:
            xSTDs = np.sqrt(
                np.bincount(imagesForClusters_contig, weights=np.square(mappedPointDifs[:, 0])) / unique_bins_count)
            ySTDs = np.sqrt(
                np.bincount(imagesForClusters_contig, weights=np.square(mappedPointDifs[:, 1])) / unique_bins_count)
            stdScores = 1 / (1 + np.sqrt(np.sqrt(np.sqrt((xSTDs + ySTDs)))) / 2)
            voteScores *= stdScores  # st.norm.pdf((xSTDs+ySTDs)/2,scale=2.5)/st.norm.pdf(0,scale=2.5)

        if returnClusters:
            outcinds = imagesForClusters.argsort()
            outclusters = np.split(all_vq_bin_ids[outcinds], np.cumsum(unique_bins_count)[:-1])
            return voteScores, final_image_IDs, unique_bins_count, outclusters, mappedPointDists, voteSums, meandists

        return voteScores, final_image_IDs, unique_bins_count
