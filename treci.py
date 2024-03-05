try:
    brojevi = []

    while (1):
        vrijednost = input()
        if (vrijednost == "Done"):
            break
        elif (not vrijednost.isnumeric()):
            raise Exception("Nisi unio broj ili 'Done'")
            break
        brojevi.append(int(vrijednost))     
        
    print(len(brojevi))
    print(max(brojevi))
    print(min(brojevi))
    print(sum(brojevi) / len(brojevi))
except Exception as e:
    print(e)


