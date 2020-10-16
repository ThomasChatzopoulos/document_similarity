from numpy.core._multiarray_umath import ndarray
from sklearn.feature_extraction.text import CountVectorizer # library for CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity # library for cosine_similarity
import numpy as np


def binFact (x): #binomal factorial (function for calculating the combination of (n k) objects), formula: div = n!/(k!(n-k)!)
    # k is always 2, so I only enter the value of n,  ie the number of files (Ν), so it is valid: (n-1)n/2
    try: # there is a division so I use try/except
        div = ((x-1)*x)/2 # caclulation of binomial factor
    except:
        print("Error in calculation") #error mesagge
    return div #rerutn the result of division

numOfFiles=0 #documents, mount of documents
print("\nWellcome to this application!\nHere you can check the similarity between documents.\n")
while numOfFiles<2: # there must be at least 2 files
    print("How many documents (>=2) do you want to check?")
    numOfFiles=int(input("")) #save the number of files

namesOfFiles=[] #Array with the file names
indexOfFiles=[] #Array with the indexes of files

for i in range(numOfFiles): #read files
    namesOfFiles.append(input("write the file " + str(i + 1) + " name (the fullname, for example: doc" + str(i + 1) + ".txt)"))
    try:
        fr=open(namesOfFiles[i],"r") #input stream
        indexOfFiles.append(fr.read().lower()) # read file, conversion into lowercase letters
        fr.close() # close file
    except:
        print("Erron on loading file") #error message

maxSimilar=binFact(numOfFiles) # number of the max most similar documents
mountOfSimilarDoc=2048 # users number for the mount of similar documents

while mountOfSimilarDoc > binFact(numOfFiles): # oso o ari8mos twn koinwn arxeiwn einai megaluteros apo (numOfFiles 2)
    mountOfSimilarDoc=int(input("\nHow many of the most similar documents do you want to see? (You can choose "+ str(maxSimilar) + " documents the most)\n"))
    if mountOfSimilarDoc <0:
        mountOfSimilarDoc=2048
        print("The number of documents must be 0 or higher!")

# convert text files into vectors number of impressions
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(indexOfFiles) # change the table form
mountOfWords=len(vectorizer.get_feature_names()) # measure the number of occurrences of each word of document
vectors=X.toarray() #vector to array

newVector=[]
for i in range(numOfFiles):
    newVector.append(vectors[i].reshape(1,mountOfWords)) # change the table form, separate line for each document

similarity: ndarray = np.zeros((numOfFiles,numOfFiles)) #Array NxN for document similarity

for i in range(numOfFiles): # filling the NxN array
    for j in range(i):      # lower triangular
        similarity[i][j]=(100*cosine_similarity(newVector[i],newVector[j])) # calculate the cosine similarity

print(similarity,"\n") # the similarity array

max=similarity[0][0] # max value of array
maxPosition=[1,1]    # position of document with me max value in the array
print("The "+str(mountOfSimilarDoc)+" most similar documents:")

for z in range(mountOfSimilarDoc): # runs for the multitude of most similar documents (Ν)
    for i in range(numOfFiles):    # runs for all the bars of array
        for j in range(i+1):       # for the lower triangular
            if max <= similarity[i][j]: # select each time the next largest number
                max=similarity[i][j] # save max value
                maxPosition = [i , j] #save the position of max value
    # print the results
    print(str(z + 1), ". document: ", namesOfFiles[maxPosition[1]], "\t with document: ", namesOfFiles[maxPosition[0]],"\t with a similarity of ", similarity[maxPosition[0], maxPosition[1]], "%")
    similarity[maxPosition[0], maxPosition[1]] = 0 # zeroing
    max = 0 # zeroing
