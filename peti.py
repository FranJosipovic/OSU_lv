fhand = open('SMSSpamCollection.txt')

dictionary = {}

total_ham_words = 0
total_hams = 0

total_spam_words = 0
total_spams = 0

total_usklicnik = 0

for line in fhand :
    line = line.rstrip()
    words = line.split()
    key = words[0]

    if(key == "ham"):
        total_hams += 1
        message = words[1:]
        total_ham_words += len(message)
    
    if(key == "spam"):
        total_spams += 1
        message = words[1:]
        total_spam_words += len(message)
        if(message[-1] == '!' or message[-1].endswith('!')):
            total_usklicnik += 1

    
print(total_ham_words / total_hams)
print(total_spam_words / total_spams)
print(total_usklicnik)

fhand.close ()