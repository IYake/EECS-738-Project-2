from hmm import load, generate, predict

def main():
    choice = None
    numWords = None
    sequence = None
    model = load("hmm.pickle")
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
            print("How many words to predict? ")
            numWords = int(input())
            predict(model,numWords,sequence)
        print("------------------------")

if __name__ == "__main__":
    main()
