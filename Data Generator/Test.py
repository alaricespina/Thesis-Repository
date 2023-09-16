arr = []

def recur(n):
    if n > 0:
        arr.append(n)
        recur(n-1)
    else:
        print(arr)
        print("End Reached")

recur(2)