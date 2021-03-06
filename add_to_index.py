from indexConstruction import indexMerger
from indexConstruction import indexConstruction
import argparse
import os
import sys
import logging
import traceback
from resources import Resource

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--FeatureFileList', help='provenance index file')
    parser.add_argument('--IndexOutputFile', help='output directory for the Index')
    parser.add_argument('--TrainedIndexParams', help='file generated by indexTraining.py')
    parser.add_argument('--CacheFolder', help='where to save all intermediate files',default='.')
    parser.add_argument('--Numshards', help='How many total shards to run (one shard per job)',default=1,type=int)
    parser.add_argument('--Shardnum', help='which shard is this?',default=-1)
    parser.add_argument('--GPUCount', help='How many GPUs to use?', default = 1,type=int)

    args = parser.parse_args()
    if args.Shardnum == -1:
        shardsToRun = range(args.Numshards)
    else:
        shardsToRun = [int(t) for t in args.Shardnum.split(',')]

    indexShardFolderNames = []
    featOffset=0
    imOffset = 0
    for shardNum in shardsToRun:
        indexConstructor = indexConstruction(indexSaveFolder=args.IndexOutputFile,cachefolder=args.CacheFolder,numshards=args.Numshards,shardnum=shardNum,featureOffset=featOffset,imageOffset=imOffset,gpuCount=args.GPUCount,featdims=64)
        if args.TrainedIndexParams is not None:
            indexparameters = open(args.TrainedIndexParams,'rb')
            indexResource = Resource('indexparameters', indexparameters.read(),'application/octet-stream')
            indexConstructor.loadIndexParameters(indexResource)
        else:
            indexConstructor.loadIndexParameters(None)
        import time
        t0 = time.time()
        indexDict,featOffset,imOffset = indexConstructor.buildIndex(args.FeatureFileList)
        if not indexDict.endswith('index.index'):
            indexDict = os.path.join(indexDict,'index.index')
        indexShardFolderNames.append(indexDict)
        t1 = time.time()
        with open('runtimeStat.txt','w') as fp:
            fp.write('runtime in seconds: '+ str(t1-t0))

    print(indexShardFolderNames)

    indexOutputFolder = os.path.dirname(args.IndexOutputFile)
    indexMerger.mergeIndexShards(indexShardFolderNames,indexConstructor.emptyIndexPath,args.IndexOutputFile)
