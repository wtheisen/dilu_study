import collections

import indexfunctions

from scoreMerge import mergeResultsMatrix

#Thiss class# produces the data needed for the Provenance Filtering JSON
#the function merge will be used to merge results when indexing is parallelized
# you can modify the class implementations to meet your needs, but function calls
# should be kept the same
class filteringResults:
    map = {}
    scores = collections.OrderedDict()
    def __init__(self):
        self.probeImage = ""
        self.I = None
        self.D = None
        self.map = {}
        self.scores = collections.OrderedDict()
        self.visData = collections.OrderedDict()
    def addScore(self,filename, score,ID=None,visData=None):
        self.scores[filename]=score
        if ID is not None:
            self.map[ID] = filename
        if visData is not None:
            self.visData[filename] = visData
    #this function merges two results
    def mergeScores(self,additionalScores,ignoreIDs = []):
        if self.I is not None and self.D is not None and additionalScores is not None and additionalScores.I is not None and additionalScores.D is not None:
            print('merging scores using I and D')
            # Merge results based on I and D matrixes (not heuristic!)
            mergedresults = mergeResultsMatrix(self.D,additionalScores.D,self.I,additionalScores.I,self.map,additionalScores.map,k=min(len(self.scores),self.I.shape[1]),numcores=12)
            self.I = mergedresults[0]
            self.D = mergedresults[1]
            self.map = mergedresults[2]
            sortedIDs, sortedVotes,maxvoteval = indexfunctions.tallyVotes(self.D, self.I)
            # voteScores = 1.0 * sortedVotes / (1.0 * np.max(sortedVotes))
            voteScores = 1.0 * sortedVotes / (maxvoteval)
            self.scores = collections.OrderedDict()
            for i in range(0, len(sortedIDs)):
                id = sortedIDs[i]
                if id not in ignoreIDs:
                    id_str = str(id)
                    if id in self.map:
                        imname = self.map[id]
                        score = voteScores[i]
                        self.addScore(imname, score, ID=id)

        elif additionalScores is None:
            #if additional scores contains nothing don't add anything!
            pass
        elif self.I is None and self.D is None and additionalScores.I is not None and additionalScores.D is not None:
            # Pushing into empty results, just populate the object with the additionalScores
            self.I = additionalScores.I
            self.D = additionalScores.D
            self.map = additionalScores.map
            self.scores = additionalScores.scores

        else:
            # Merge in a heuristic way
            self.scores.update(additionalScores.scores)
            if additionalScores.visData is not None:
                self.visData.update(additionalScores.visData)
                #sortinds = np.array(self.scores.values()).argsort()
                #vd = self.visData.copy()
                #self.visData.clear()
                #for v in np.array(list(vd.keys())).argsort()[::-1]:
                #   self.visData[v] = vd[v]
            sortedscores = collections.OrderedDict(sorted(self.scores.items(), key=lambda x: x[1], reverse=True))
            self.scores = sortedscores
        for id in ignoreIDs:
            if id in self.scores:
                del self.scores[id]
    # this function merges two results
    def dictSort(self, additionalScores):
        od = collections.OrderedDict(sorted(self.scores.items(), key=lambda x: x[1], reverse=True))
        self.scores.update(additionalScores.scores)
        sortedscores = collections.OrderedDict(sorted(self.scores.items(), key=lambda x: x[1], reverse=True))
        self.scores = sortedscores

    #Once scores are merged together, at most "numberOfResultsToRetrieve" will be retained
    def pairDownResults(self,numberOfResultsToRetrieve):
        numberOfResultsToRetrieve = int(numberOfResultsToRetrieve)
        if len(self.scores) > numberOfResultsToRetrieve:
            newscores = collections.OrderedDict(
                sorted(self.scores.items(), key=lambda x: x[1], reverse=True)[:numberOfResultsToRetrieve])
            self.scores = newscores
    def normalizeResults(self):
        maxVal = self.scores[list(self.scores.keys())[0]]
        for s in self.scores:
            self.scores[s] = self.scores[s]/maxVal

