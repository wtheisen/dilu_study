
def visualizeVotes_toRank(mappedVotes, mappedPoints, mappedMatchScores, mappedDists, mappedQSizes, imagesForClusters,
                          query_centroids, imageIDs, imageScores, probename, IDToImage, d, rank=100, outdir=''):
    sortinds = imageScores.argsort()[::-1]
    sortedScores = imageScores[sortinds]
    sortedScores = (sortedScores - sortedScores.min()) / (sortedScores.max() - sortedScores.min())
    sortedIDs = imageIDs[sortinds]
    probeimg = cv2.imread(d[probename])
    outputdir = os.path.join(outdir, probename.split('.')[0])
    if not os.path.isdir(outputdir):
        os.makedirs(outputdir)
    print('visualizing...')
    bar = progressbar.ProgressBar()
    for r in bar(range(min(rank, len(sortedIDs)))):
        try:
            resultID = sortedIDs[r]
            resultname = IDToImage[str(resultID)]
            nscore = sortedScores[r]
            resultVis = visualizeVotes_forImage(mappedVotes, mappedPoints, mappedMatchScores, mappedDists, mappedQSizes,
                                                imagesForClusters, resultID, IDToImage, d, nscore=1)
            csize = len(mappedMatchScores[imagesForClusters == resultID])
            centroid = query_centroids[imagesForClusters == resultID][0]
            rad = min(15, int(max(probeimg.shape[0], probeimg.shape[1]) / 25))
            qimg_withCenter = cv2.circle(probeimg.copy(), (int(centroid[0]), int(centroid[1])), rad, (0, 0, 255), -1)
            savename = 'r' + str(r).zfill(3) + '_' + str(csize) + '_' + resultname
            qsavename = 'query_' + savename
            cv2.imwrite(os.path.join(outputdir, savename + '.png'), resultVis)
            cv2.imwrite(os.path.join(outputdir, qsavename + '.png'), qimg_withCenter)
            origdir = os.path.join(outputdir, 'originalImages')
            if not os.path.isdir(origdir):
                os.makedirs(origdir)
            cv2.imwrite(os.path.join(origdir, savename + '.png'), cv2.imread(d[resultname]))

        except:
            pass

def visualizeVotes_forImage(mappedVotes, mappedPoints, mappedMatchScores, mappedDists, mappedQSizes, imagesForClusters,
                            imageID, IDToImage, d, nscore=1):
    imageName = IDToImage[str(imageID)]
    imageDir = d[imageName]
    img = cv2.imread(imageDir)
    votemap = np.zeros(img.shape[:2], np.double)
    pdffunc = st.norm.pdf
    stepsPerVote = 20
    useInds = imagesForClusters == imageID
    votes = mappedVotes[useInds]
    voteScores = mappedMatchScores[useInds]
    matchDists = 1 / voteScores - 1
    matchDists = matchDists / matchDists.max()
    points = mappedPoints[useInds]
    voteDists = mappedDists[useInds]  # The radiuses
    qwindowSizes = mappedQSizes[useInds]
    windowsize = qwindowSizes[0]
    for i in range(len(votes)):
        radius = voteDists[i] * 2
        stepsize = 1
        xrng = np.arange(stepsize, radius, stepsize)
        pf = pdffunc(xrng / windowsize, scale=1.5) / pdffunc(0, scale=1.5)
        pf = (pf - pf.min()) / (1 - pf.min())
        linkern = np.concatenate((pf[::-1], [1], pf)).reshape((1, -1))
        kern = linkern.T * linkern
        # plt.imshow(kern)
        startx = int(votes[i][0] - kern.shape[1] / 2)
        starty = int(votes[i][1] - kern.shape[0] / 2)
        endx = startx + kern.shape[1]
        endy = starty + kern.shape[0]
        neededShape = votemap[starty:endy, startx:endx].shape
        votemap[starty:endy, startx:endx] += (kern * voteScores[i])[:neededShape[0], :neededShape[1]]

    cmap = plt.cm.jet
    # votemap = votemap.max()-votemap
    norm = plt.Normalize(vmin=votemap.min(), vmax=votemap.max())
    votemap *= nscore
    votemap = votemap.max() - votemap
    votemap = votemap - votemap.min()
    cmapimg = (cmap(np.square(norm(votemap)))[:, :, :-1] * 255).astype(np.uint8)
    overlay = cmapimg.copy()  # cv2.cvtColor(cmapimg.copy(),cv2.COLOR_BGR2RGB)
    output = img.copy()  # cv2.cvtColor(img.copy(),cv2.COLOR_BGR2RGB)
    cv2.addWeighted(overlay, .5, img, .5, 0, output)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    return output
    plt.imshow(output)
    # vmags = np.linalg.norm(votes-points,axis=1)
    # print(vmags)
    # plt.scatter(voteDists,voteScores)


