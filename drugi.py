try:
    vrijednost = input("unesi broj izmedu 0.0 i 1.0: ")
    if(vrijednost == ''):
        raise Exception("nisi unio broj uopce")
    broj = float(vrijednost)
    if(broj > 1.0 or broj < 0.0):
        raise Exception("nisi unio dobar broj")
    
    if (broj >= 0.9):
        print("A")
    elif (broj >= 0.8):
        print("B")
    elif (broj >= 0.7):
        print("C")
    elif (broj >= 0.6):
        print("D")
    else:
        print("F")
except Exception as e:
    print(e)

