import pandas as pd
import numpy as np
import glob
import functools

def read_multiple_csv_and_concat(file_location_pattern_and_name_pattern):

    """
    EXAMPLE: file_location_pattern_and_name_pattern = '../../data/01_raw/Kickstarter_201*/Kickstarter*'
    """

    files = glob.glob(file_location_pattern_and_name_pattern)
    print(files)

#     a = [0,"['Grilled Cheese BLT']","['Prep time:', ' ', '10 minutes']","['Cook time:', ' ', '10 minutes']","['Yield:', ' ', '4 sandwiches']","['Filed under:', ' ', 'Dinner', 'Lunch', 'Sandwich', 'Favorite Summer', 'Quick and Easy', 'Bacon', 'Grilled Cheese Sandwich', 'Lettuce', 'Pork', 'Tomato']","['\n                        ', 'Ingredients', '\n\n                        ', '\n', '8 slices sourdough bread', '\n', '4 tablespoon unsalted butter, at room temperature', '\n', '8 ounces (2 cups) shredded cheddar cheese', '\n', '2 slicing tomatoes (such as beefsteak, Brandywine, or Cherokee purple), sliced 1/4-inch thick', '\n', '8 to 12 slices ', 'cooked bacon', '\n', '12 leaves butterhead or other crispy lettuce', '\n', 'Kosher salt and black pepper', '\n', '\n\n\t\t\t\t\t\t\n                                              ']","['\n\n                    ', '\n                      ', '\n        ', '\n            ', '\n            Save It\n        ', '\n    ', '                      ', 'Print', '                    ', '\n\n                    ', 'Grilled Cheese BLT Recipe', '\n\n                                          ', '\n                        ', '\n                          ', 'Prep time:', ' ', '10 minutes', 'Cook time:', ' ', '10 minutes', 'Yield:', ' ', '4 sandwiches', '                        ', '\n                      ', '\n                    \n                    \n                                          ', '\n                        ', 'Ingredients', '\n\n                        ', '\n', '8 slices sourdough bread', '\n', '4 tablespoon unsalted butter, at room temperature', '\n', '8 ounces (2 cups) shredded cheddar cheese', '\n', '2 slicing tomatoes (such as beefsteak, Brandywine, or Cherokee purple), sliced 1/4-inch thick', '\n', '8 to 12 slices ', 'cooked bacon', '\n', '12 leaves butterhead or other crispy lettuce', '\n', 'Kosher salt and black pepper', '\n', '\n\n\t\t\t\t\t\t\n                                              ', '\n                    \n                    \n                                        \n                                            ', '\n                        \n                    ', '\n                      ', 'Method', 'Hide Photos', '\n                      ', '1', ' ', 'Melt the cheese: ', 'For each sandwich, spread 1 tablespoon butter on one side of two pieces of bread. Place the bread buttered-side down in a large skillet or on a griddle over medium-low heat. Sprinkle 2 ounces (1/2 cup) shredded cheese evenly over both slices. Cook until the cheese has melted and the bread is crisp, about 2 minutes.', '\n', '\n', '2 Assemble the sandwich', ': Remove from the skillet. Layer one side with 2 to 3 pieces of bacon, the slices from half a tomato seasoned with some salt and pepper, and 3 leaves of lettuce. Close the sandwich. Repeat for remaining sandwiches. Slice in half and eat warm.', '\n', '\n', '\n                    ', '\n\n                    ', '\n                      ', 'Hello!', ' All photos and content are copyright protected. Please do not use our photos without prior written permission. ', 'Thank you!', '\n                    ', '\n\n                    ', '\n                      ', '\n        ', '\n            ', '\n            Save It\n        ', '\n    ', '                      ', 'Print', '                    ', '\n\n                  ']","['by   ', '   ', 'Aaron Hutcherson', 'August 27, 2019']","['<link rel=""canonical"" href=""https://www.simplyrecipes.com/recipes/grilled_cheese_blt/"">']"]

#     myvar = pd.Series(a)

    li = []
    for filename in files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)

    return frame

def singularize(word):
    """
    A poor replacement for the pattern.en singularize function, but ok for now.
    """

    units = {
        "cups": u"cup",
        "tablespoons": u"tablespoon",
        "teaspoons": u"teaspoon",
        "pounds": u"pound",
        "ounces": u"ounce",
        "cloves": u"clove",
        "sprigs": u"sprig",
        "pinches": u"pinch",
        "bunches": u"bunch",
        "slices": u"slice",
        "grams": u"gram",
        "heads": u"head",
        "quarts": u"quart",
        "stalks": u"stalk",
        "pints": u"pint",
        "pieces": u"piece",
        "sticks": u"stick",
        "dashes": u"dash",
        "fillets": u"fillet",
        "cans": u"can",
        "ears": u"ear",
        "packages": u"package",
        "strips": u"strip",
        "bulbs": u"bulb",
        "bottles": u"bottle"
    }

    if word in units.keys():
        return units[word]
    else:
        return word

def word2features(doc, i):
    word = doc[i][0]
    postag = doc[i][1]

    # Common features for all words
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'postag=' + postag
    ]

    # Features for words that are not
    # at the beginning of a document
    if i > 0:
        word1 = doc[i-1][0]
        postag1 = doc[i-1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:postag=' + postag1
        ])
    else:
        # Indicate that it is the 'beginning of a document'
        features.append('BOS')

    # Features for words that are not
    # at the end of a document
    if i < len(doc)-1:
        word1 = doc[i+1][0]
        postag1 = doc[i+1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:postag=' + postag1
        ])
    else:
        # Indicate that it is the 'end of a document'
        features.append('EOS')

    return features

# A function for extracting features in documents
def extract_features(doc):
    return [word2features(doc, i) for i in range(len(doc))]

# A function fo generating the list of labels for each document
def get_labels(doc):
    return [label for (token, postag, label) in doc]
