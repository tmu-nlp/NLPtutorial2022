from nltk.tree import Tree

def print_tree(file):
    with open(file, "r") as data:
        for line in data:
            t = Tree.fromstring(line)
            t.pretty_print()

print_tree("wiki-ans")
