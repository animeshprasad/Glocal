from gensim.summarization.keywords import _set_graph_edges, _build_graph, \
    _clean_text_by_word, _tokenize_by_word, _get_words_for_graph
from gensim.summarization.textcleaner import replace_with_separator, tokenize,\
    join_words, merge_syntactic_units
from gensim.parsing.preprocessing import strip_tags, strip_punctuation, \
    strip_multiple_whitespaces, strip_numeric,remove_stopwords, strip_short
from gensim import utils
import re
from gensim.summarization.graph import Graph


try:
    from pattern.en import tag
    HAS_PATTERN = True
except ImportError:
    HAS_PATTERN = False

AB_ACRONYM_LETTERS = re.compile(r'([a-zA-Z])\.([a-zA-Z])\.', re.UNICODE)

DEFAULT_FILTERS = [
    lambda x: x.lower(), strip_tags, strip_punctuation,
    strip_multiple_whitespaces, strip_numeric,
    remove_stopwords, strip_short
]

NO_LOWER_FILTERS = [
    strip_tags, strip_punctuation,
    strip_multiple_whitespaces, strip_numeric,
    remove_stopwords, strip_short
]


def preprocess_string_nolower(s, filters=NO_LOWER_FILTERS):
    s = utils.to_unicode(s)
    for f in filters:
        s = f(s)
    return s.split()


def preprocess_string(s, filters=DEFAULT_FILTERS):
    s = utils.to_unicode(s)
    for f in filters:
        s = f(s)
    return s.split()


def preprocess_documents_no_stem(docs):
    return [preprocess_string(d) for d in docs]


def _clean_text_by_word_no_stem(text, deacc=True):
    text_without_acronyms = replace_with_separator(text, "", [AB_ACRONYM_LETTERS])
    original_words = list(tokenize(text_without_acronyms, to_lower=True, deacc=deacc))
    filtered_words = [join_words(word_list, "") for word_list in preprocess_documents_no_stem(original_words)]
    if HAS_PATTERN:
        tags = tag(join_words(original_words))  # tag needs the context of the words in the text
    else:
        tags = None
    units = merge_syntactic_units(original_words, filtered_words, tags)
    return {unit.text: unit for unit in units}


def get_graph(text, stem=False):
    tokens = _clean_text_by_word_no_stem(text)
    if stem:
        tokens = _clean_text_by_word(text)
    split_text = list(_tokenize_by_word(text))
    graph = _build_graph(_get_words_for_graph(tokens))
    _set_graph_edges(graph, tokens, split_text)

    return graph

def get_graph2(text, stem=False):
    tokens = _clean_text_by_word_no_stem(text)
    if stem:
        tokens = _clean_text_by_word(text)


    graph = Graph()
    for item in tokens:
        if not graph.has_node(item):
            graph.add_node(item)


    split_text = list(_tokenize_by_word(text))

    print (split_text)
    input()

    # graph = _build_graph(_get_words_for_graph(tokens))
    _set_graph_edges(graph, tokens, split_text)

    return graph


if __name__=="__main__":
    print (get_graph('substitution'))
    print(get_graph('substitution'), True)