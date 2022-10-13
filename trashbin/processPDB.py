import getopt
import os
import sys
import Bio.SeqUtils


def extract_relevant_lines(filename):
    inputfile = open(filename, "r")
    relvt_lines = line_filter(inputfile.readlines())
    return relvt_lines


def line_filter(inputLines):
    atomic_pos_keywrd  = ["ATOM", " CA ", " A "]
    return [line for line in inputLines if
            all(word in line for word in atomic_pos_keywrd)]

def process_lines(inputLines):
    sequence = ""
    location = ""
    three_let_code = [line[17:20] for line in inputLines]
    z_pos = [line[47:54] for line in inputLines]

    for codon in three_let_code:
        sequence += Bio.SeqUtils.IUPACData.protein_letters_3to1[codon.capitalize()]

    for number in z_pos:
        value = float(number)
        if value > 10:
            location += 'O'
        elif value > -10:
            location += 'M'
        else:
            location += 'I'

    return sequence, location

if __name__ == "__main__":
    # Command line processing ---------------------------------------------------------------------
    filepath = ""
    argumentList = sys.argv[1:]
    try:
        filepath = argumentList[0]

    except IndexError as err:
        print("Command failed...USAGE: ")
        print("python RandomGenerator.py <filename.pdb>")

    # -------- Command line processing complete ---------------------------------------------------

    relevant_lines = extract_relevant_lines(filepath)
    print(process_lines(relevant_lines))