# Importing Colours from a randomly generated Colours file
file = open('colours.txt', 'r')
# Converting to string
temp = ''.join(file.read())
file.close()
# remove trailing whitespaces
temp.rstrip()
# split on whitespaces
t = temp.split()

# colour list
COLOURS = []
file = open('colours-edit.txt', 'w')
for i in t:
    i = '"#' + i + '"'
    COLOURS.append(i)
    file.write(i + ', ')

file.close()
