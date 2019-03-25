
from fwd_bck import load, generate

def main():
    numWords = 15

    model = load("hmm.pickle")
    generate(model,numWords)

if __name__ == "__main__":
    main()
