import sys

from numpy.core._multiarray_umath import ndarray
from sklearn.feature_extraction.text import CountVectorizer  # library for CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity  # library for cosine_similarity
import numpy as np
from prettytable import PrettyTable


def binFact(x):  # binomal factorial (function for calculating the combination of (n k) objects), formula: div = n!/(k!(n-k)!)
    # k is always 2, so I only enter the value of n,  ie the number of files (Ν), so it is valid: (n-1)n/2
    try:  # there is a division so I use try/except
        div = ((x - 1) * x) / 2  # caclulation of binomial factor
    except:
        print("Error in calculation")  # error mesagge
    return div  # rerutn the result of division


numOfFiles = 0  # documents, mount of documents
namesOfFiles = []  # Array with the file names
indexOfFiles = []  # Array with the indexes of files
name = ""  # input file name
maxSimilar = 0  # number of the max most similar documents
mountOfSimilarDoc = -1  # users number for the mount of similar document
newVector = []
i = 0  # counter

print("\nWith this application you can check the similarity between documents.\nHow many documents (>=2) do you want to check?")
while numOfFiles < 2:  # there must be at least 2 files
    try:
        numOfFiles = int(input(""))  # save the number of files
    except ValueError:
        print("Please type an integer")

while i < numOfFiles:  # read files
    name = input("write the file " + str(i + 1) + " name (the fullname, for example: doc" + str(i + 1) + ".txt)")
    namesOfFiles.insert(i, name)
    try:
        fr = open("documents/" + namesOfFiles[i], "r")  # input stream
        indexOfFiles.append(fr.read().lower())  # read file, conversion into lowercase letters
        fr.close()  # close stream
        i = i+1
    except OSError:
        print("Loading error with: '" + name + "' . Try again.")  # error message

maxSimilar = binFact(numOfFiles)  # number of the max most similar documents
print("\nHow many of the most similar documents do you want to see? (You can select a maximum of " + str(int(maxSimilar)) + ")\n")
while (mountOfSimilarDoc < 0) :
    try:
        mountOfSimilarDoc = int(input(""))  # save the number of files
        if mountOfSimilarDoc < 0:
            print("The number of documents must be 0 or higher!")
            mountOfSimilarDoc = -1
    except ValueError:
        print("Please type an integer")

# convert text files into vectors number of impressions
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(indexOfFiles)  # change the table form
mountOfWords = len(vectorizer.get_feature_names())  # measure the number of occurrences of each word of document
vectors = X.toarray()  # vector to array

for i in range(numOfFiles):
    newVector.append(vectors[i].reshape(1, mountOfWords))  # change the table form, separate line for each document

similarity: ndarray = np.zeros((numOfFiles, numOfFiles))  # Array NxN for document similarity

for i in range(numOfFiles):  # filling the NxN array
    for j in range(i):  # lower triangular
        similarity[i][j] = (100 * cosine_similarity(newVector[i], newVector[j]))  # calculate the cosine similarity

# print(similarity, "\n")  # the similarity array

max = similarity[0][0]  # max value of array
maxPosition = [1, 1]  # position of document with me max value in the array
r = PrettyTable(['No', 'First Document', 'Second Document', 'Similarity (%)'])
print("The " + str(mountOfSimilarDoc) + " most similar documents:")

for z in range(mountOfSimilarDoc):  # runs for the multitude of most similar documents (Ν)
    for i in range(numOfFiles):  # runs for all the bars of array
        for j in range(i + 1):  # for the lower triangular
            if max <= similarity[i][j]:  # select each time the next largest number
                max = similarity[i][j]  # save max value
                maxPosition = [i, j]  # save the position of max value
    # print the results
    r.add_row([str(z + 1), namesOfFiles[maxPosition[1]], namesOfFiles[maxPosition[0]], round(similarity[maxPosition[0], maxPosition[1]], 3)])
    # print(str(z + 1), ". ", namesOfFiles[maxPosition[1]], " - ", namesOfFiles[maxPosition[0]],
    #       " similarity of ", similarity[maxPosition[0], maxPosition[1]], "%")
    similarity[maxPosition[0], maxPosition[1]] = 0  # zeroing
    max = 0  # zeroing
print(r)
