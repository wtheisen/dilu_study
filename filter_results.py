
def filter_query_results(I, D, ID_to_path, use_nhs=True, result_num=500):
    if use_nhs:
        sortedIDs, sortedVotes,maxvoteval = NeedleInHaystack.nhscore(I, D,
                image_level_IDs, imagesizes, allMetaDatabase, allMetaQuery,
                visualize=False, numberOfResultsToRetrieve=result_num)
    else:
        sortedIDs, sortedVotes, maxvoteval = indexfunctions.tallyVotes(D, I, image_level_IDs, numcores=1)

    resultScores = filteringResults()

    for i in range(0, min(len(sortedIDs), result_num * 10)):
        id = sortedIDs[i]
        id_str = str(int(id))

        if id_str in ID_to_path:
            img_path = ID_to_path[id_str]
            score = sortedVotes[i]
            resultScores.addScore(img_path, score, ID=id)
    resultScores.pairDownResults(result_num)

    return resultScores
