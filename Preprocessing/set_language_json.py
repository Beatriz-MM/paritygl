# Author: Beatriz Molina Muñiz (GitHub: @Beatriz-MM)
# Last modified: 14/04/2025
# Description: Detects and classifies the language of comments in JSON files as Galician (gl), Spanish (es), or undetermined (cpo)
# Python version: 3.10.12

import json
import re

VOWELS = 'aeiouáéíóúAEIOUÁÉÍÓÚ'

GALICIAN_PRONOUNS = ['eu', 'ti', 'el', 'ela', 'nós', 'vós', 'eles', 'elas']
SPANISH_PRONOUNS = ['yo', 'tu', 'el', 'ella', 'nosotros', 'vosotros', 'ellos', 'ellas']

GALICIAN_POSSESSIVES = ['meu', 'teu', 'del', 'dela', 'noso', 'voso', 'deles', 'delas']
SPANISH_POSSESSIVES = ['mío', 'mio', 'tuyo', 'suyo', 'nuestro', 'vuestro']

GALICIAN_ARTICLES = ['unha', 'unhas', 'uns', 'a', 'o', 'as', 'os', 'ás', 'ós', 'aos']
SPANISH_ARTICLES = ['una', 'uno', 'unas', 'unos', 'la', 'lo', 'el', 'los', 'las']

SPANISH_POSSESSIVE_ARTICLES = ['de el', 'de ella', 'de ellos', 'de ellas', 'a las', 'a los', 'en el']

GALICIAN_KEYWORDS = ['é', 'sodes', 'máis', 'mais', 'vostede', 'grazas', 'xa', 'ou', 'non', 'deus', 'aínda', 'ainda']
SPANISH_KEYWORDS = ['eres', 'sois' 'más', 'mas', 'usted', 'ya', 'dios', 'aún', 'aun']

COMPOUND_VERBS = ['he', 'había', 'habia', 'habré', 'habre', 'habría', 'habria', 'has', 'habías','habias', 'habrás', 'habras', 'habrías', 'habrias',
                      'ha', 'habrá', 'habra', 'hemos', 'habíamos', 'habiamos', 'habremos', 'habríamos', 'habriamos', 'habéis', 'habeis', 'habíais', 'habiais',
                      'habréis', 'habreis', 'habríais', 'habriais', 'han', 'habían', 'habian', 'habrán', 'habran', 'habrían', 'habrian']


def open_file(comments_file):
    with open(comments_file, 'r', encoding='utf-8') as file:
        return json.load(file)


def detect_nh(text):
    """
    Detects the presence of the 'nh' digraph in Galician text.

    Args:
        text (str): The input text to search for 'nh' digraph.
    Returns:
        bool: True if 'nh' is found in the text, otherwise False.
    """ 
    pattern = r'[{}]nh[{}]'.format(VOWELS, VOWELS)
    return bool(re.search(pattern, text))

def detect_y(text):
    """
   Detects the presence of the letter 'y' used as a conjunction.

    Args:
        text (str): The input text to search for the conjunction 'y' or 'Y'.
    Returns:
        bool: True if 'y' or 'Y' is found in the text, otherwise False.
    """ 
    pattern = r'\b[yY]\b|[{}][yY][{}]|\b[yY][^{}aeiouáéíóúAEIOUÁÉÍÓÚ]|\b[yY]$'.format(VOWELS, VOWELS, VOWELS)
    return bool(re.search(pattern, text))

# Conteo del uso de 'x' y 'j' en el text
def count_x_j(text):
    """
    Counts the occurrences of the letters 'x' and 'j' in the text.

    Args:
        text (str): The input text to search for the letters 'x' and 'j'.
    Returns:
        tuple: A tuple containing the count of 'x' and the count of 'j' in the text.
    """ 
    count_x = len(re.findall(r'[xX]', text))
    count_j = len(re.findall(r'[jJ]', text))
    return count_x, count_j

def contains_che_te(text):
    """
     Detects the presence of the Galician clitic "che" or "te".

    Args:
        text (str): The input text to search for "che" or "te".
    Returns:
        bool: True if "che" or "te" is found in the text, otherwise False.
    """ 
    pattern = r'\b(che|te)\b|[{}](che|te)[{}]'.format(VOWELS, VOWELS)
    return bool(re.search(pattern, text))

def contains_compound_verbs(text, compound_verbs):
    """
    Detects the presence of compound verbs.

    Args:
        text (str): The input text to search for compound verbs.
        compound_verbs (list): A list of compound verbs to search for in the text.
    Returns:
        bool: True if any compound verb is found, otherwise False.
    """ 
    for compound_verb in compound_verbs:
        pattern = r'\b{}\s+\w+[oO]\b'.format(re.escape(compound_verb))
        if re.search(pattern, text, re.IGNORECASE):  
            return True
    return False

def contains_key_elements(text, keys):
    """
    Checks if any of the provided keywords are present in the text.

    Args:
        text (str): The input text to search for keywords.
        keys (list): A list of keywords to search for in the text.
    Returns:
        bool: True if any keyword is found, otherwise False.
    """ 
    for key in keys:
        pattern = r'\b{}\b'.format(re.escape(key))
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def detect_language(comments_file, output_file):
    """
    Analyzes a set of comments, detects whether they are in Galician or Spanish, and classifies them accordingly.

    Args:
        comments_file (str): The path to the JSON file containing comments.
        output_file (str): The path to save the processed comments with language labels.
    Returns:
        None: Saves the results directly to a new JSON file.
    """ 
    comments = open_file(comments_file)
    comments_with_language = []

    for comment in comments:
        text = comment.get('text', '').lower()

        galician_elements = sum([
            contains_key_elements(text, GALICIAN_PRONOUNS),
            contains_key_elements(text, GALICIAN_POSSESSIVES),
            contains_key_elements(text, GALICIAN_ARTICLES),
            contains_key_elements(text, GALICIAN_KEYWORDS),
            detect_nh(text),
            contains_che_te(text),
            (1 if count_x_j(text)[0] > count_x_j(text)[1] else 0)
        ])

        spanish_elements = sum([
            contains_key_elements(text, SPANISH_PRONOUNS),
            contains_key_elements(text, SPANISH_POSSESSIVES),
            contains_key_elements(text, SPANISH_ARTICLES),
            contains_key_elements(text, SPANISH_KEYWORDS),
            detect_y(text),
            contains_compound_verbs(text, COMPOUND_VERBS),
            (1 if count_x_j(text)[1] > count_x_j(text)[0] else 0)
        ])

        if galician_elements > spanish_elements:
            comment['language'] = 'gl'
        elif spanish_elements > galician_elements:
            comment['language'] = 'es'
        else:
            comment['language'] = 'cpo'


        reordered_comment = {
            "postUrl": comment["postUrl"],
            "id": comment["id"],
            "language": comment["language"],
            "text": comment["text"]
        }

        comments_with_language.append(reordered_comment)

    with open(output_file, 'w', encoding='utf-8') as file_out:
        json.dump(comments_with_language, file_out, ensure_ascii=False, indent=4)

# IMPORTANT: Specify the path to the input JSON file and the output file name
comments_file = ""
output_file = ""
detect_language(comments_file, output_file)