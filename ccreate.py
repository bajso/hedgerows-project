# Importing Colours from a randomly generated Colours file

COLOURS = []


def create_colours():
    f = open('colours_raw.txt', 'r')
    # Converting to string
    temp = ''.join(f.read())
    f.close()
    # remove trailing whitespaces
    temp.rstrip()
    # split on whitespaces
    t = temp.split()

    # populate list and output txt
    f = open('colours.txt', 'w')
    for i in t:
        i = '#' + i
        COLOURS.append(i)
        f.write(i + ', ')

    f.close()

    return COLOURS
