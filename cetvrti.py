fhand = open('song.txt')

dictionary = {}

for line in fhand :
    line = line.rstrip()
    print (line)
    words = line.split()
    for word in words:
        if(word in dictionary.keys()):
            dictionary[word] += 1
        else:
            dictionary[word] = 1

count = 0

for item in dictionary:
    if(dictionary[item] == 1):
        count += 1
        print(item)

print(count)
fhand.close ()