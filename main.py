
from fwd_bck import load, generate, predict

def main():
    choice = None
    numWords = None
    sequence = None
    model = load("model1.pickle")
    while (choice != "3"):
        print("Choose an option:")
        print("1) Generate text")
        print("2) Predict text from sequence of words")
        print("3) Quit")
        print("Choice: ")
        choice = input()
        
        if (choice == "1"):
            print("How many words to generate? ")
            numWords = int(input())
            generate(model,numWords)
        if (choice == "2"):
            print("Enter your sequence of words: ")
            sequence = tuple(input().split())
            predict(model,20,sequence)
        print("------------------------")
    
if __name__ == "__main__":
    main()
    
    