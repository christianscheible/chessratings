"""Utilities for chess experiment

Author: Christian Scheible <scheibcn@ims.uni-stuttgart.de>

"""

from string import punctuation
from collections import Counter, defaultdict
import pgn
from sys import stderr
import nltk
import re
from scipy.stats import mstats
import sklearn

# NLTK tokenizers and stemmers
tokenizer = nltk.tokenize.WordPunctTokenizer()
stemmer = nltk.stem.snowball.SnowballStemmer("english")

# patterns for feature extractio
stopset = set(["the", "is", "a", "of", "in", "are", "", "be", "to", "an"] + list(punctuation)) - set(["?", "!"])
square_pattern = re.compile("[a-f][1-8]")
move_pattern = re.compile("[kqbnr][x]?[a-f][1-8][\+\#]?")


def tokenize_annotations(annotation, binarize=True, 
                         nostop=False, lower=True, stem=False, 
                         special_features=False,
                         class_feature=False,
                         bigrams=False):
    """Tokenization and feature extraction

    Options:
      - binarize: use binary features rather than counts?
      - nostop: remove stopwords?
      - lower: lowecase tokens?
      - stem: also add stemmed words?
      - special_features: chess terminology features?
      - class_feature: add bias?
      - bigrams: also extract bigrams?
    """

    if not type(annotation) == str:
        annotation = " ".join(annotation)
        

    # different possibilities to tokenize
    tokens = re.split("[\\s\.\,]+", annotation)
    
    # lowercasing
    if lower:
        tokens = [t.lower() for t in tokens]


    # stemming
    if stem:
        # adding stems to the tokens
        tokens.extend([stemmer.stem(t) for t in tokens])
        

    if nostop:
        tokens = [t for t in tokens if t not in stopset]

    if bigrams:
        n = 2
        tokens.extend([" ".join(tokens[i:i+n]) for i in xrange(len(tokens)-n+1)]) 

    if special_features:
        # numbers
        #numbers = ["SPECIAL:num" for t in tokens if t.isdigit()]
        #tokens.extend(set(numbers))
        
        # files
        #files = ["SPECIAL:file" for t in tokens if "file" in t]
        #files = [t for t in tokens if "file" not in t]
        if any("file" in t for t in tokens):
            tokens.append("SPECIAL:file")

        # squares
        if any(square_pattern.match(t) != None for t in tokens):
            tokens.append("SPECIAL:square")

        # moves
        if any(move_pattern.match(t) != None for t in tokens):
            tokens.append("SPECIAL:move")

        #tokens.extend(files)
        
        pass
    
    if class_feature:
        tokens.append("<CLASS>")
    
    if binarize:
        tokens = set(tokens)


    return tokens


def filter_dicts(dict_list, min_df=2):
    """Remove infrequent features

    dict_list: list of feature dictionaries
    min_df: minimum frequency (default: 2)
    """

    overall = defaultdict(int)
    for d in dict_list:
        for k,v in d.iteritems():
            overall[k] += v
    
    for d in dict_list:
        for k in d.keys():
            if overall[k] < min_df:
                del d[k]

def write_svm_features(clf, vectorizer, round=1, filename="features"):
    """ Write SVM feature coefficients to file
    
    clf: sklearn classifier
    vectorizer: sklearn feature vectorizer (for reverse lookup)
    round: cross-validation round
    filename: output file name
    """

    f = open("%s-round%d.txt" % (filename, round), "w")
    weight_feature_pairs = zip(clf.coef_.tolist()[0], vectorizer.feature_names_)
    weight_feature_pairs.sort(key=lambda x:abs(x[0]), reverse=True)
    for weight, word in weight_feature_pairs:
        f.write("%s\t%g\n" % (word, weight))
    f.close()


def bineval(ratings, gold, pred, increment=50, bins = [0, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800]):
    """Evaluation split by rating bins

    gold: true labels
    pred: predicted labels
    increment: granularity parameter
    bins: pre-determined rating bins
    """

    pw_accs = [(r, int(b==c)) for r,b,c in sorted(zip(ratings, gold, pred))]
    
    binned_accs = []
    current_acc = 0
    denom = 0
    bin_id = 0
    
    for r, acc in pw_accs:
        while r > bins[bin_id] + increment:
            bin_id += 1
            if denom == 0:
                binned_accs.append(float("NaN"))
            else:
                binned_accs.append(current_acc/float(denom))
            current_acc = 0
            denom = 0
        current_acc += acc
        denom += 1

    if denom == 0:
        binned_accs.append(float("NaN"))
    else:
        binned_accs.append(current_acc/float(denom))

    for b, acc in zip(bins, binned_accs):
        print b, acc
    
class MasterSimilarity():
    """Computes similarity between amateur commentary vectors and master commentary vectors"""

    def __init__(self, filenames, vectorizer, binarize, nostop, bigrams, special_features):
        """ Set up extractor

        - filenames: list of files with master games
        - vectorizer: sklearn feature vectorizer
        - binarize: binary features rather than counts?
        - nostop: remove stopwords?
        - bigrams: also extract bigrams?
        - special_features: chess terminology features?
        """

        # copy parameters
        self.filenames = filenames
        self.vectorizer = vectorizer
        self.binarize = binarize
        self.nostop = nostop
        self.bigrams = bigrams
        self.special_features = special_features

        # data structures
        self.games = []
        self.annotations = []
        self.tokens = []
        

        # processing
        print "  Reading files"
        self.read_files()
        print "  Tokenizing"
        self.extract_annotations()
        self.tokenize()
        
        print "  Counting"
        self.token_dicts = [dict(Counter(tl)) for tl in self.tokens]

        print "  Vectorizing"
        self.vectors = vectorizer.transform(self.token_dicts)
        print "  Normalizing"
        sklearn.preprocessing.normalize(self.vectors, norm='l2', axis=1, copy=False)

        self.centroid = self.vectors.mean(axis=0)
        
    def get_mean_sim(self, fv):
        "Computes the mean similarity of a feature vector to all games in this database"
        return fv.dot(self.vectors.T).mean(axis=1)

    def get_centroid_sim(self, fv):
        "Computes the similarity of a feature vector to the centroid"
        return fv.dot(self.centroid.T)
    
        
    def tokenize(self):
        """Tokenize all games"""
        self.tokens = [tokenize_annotations(a, binarize=self.binarize, nostop=self.nostop, bigrams=self.bigrams, special_features=self.special_features) for a in self.annotations]
        
        
    def read_files(self):
        """Read games from specified files"""
        for f in self.filenames:
            self.games.extend(pgn.loads(open(f).read()))

    def extract_annotations(self, min_annot=3):
        """Extract text from all games"""
        for g in self.games:
            annotation_list = [move.strip("{}") for move in g.moves if move.strip().startswith("{")]
            if len(annotation_list) < min_annot:
                continue

            annotation = " ".join(annotation_list)
            self.annotations.append(annotation)
            
        
    
class LexiconFeatures():
    """Chess-related lexical features"""

    def __init__(self, filename):
        """Initialize with dictionary file (filename)"""

        self.term_dict = {}
        for line in open(filename):
            if line.startswith("#"):
                continue

            #print line
            word, w_type = line.strip().split("\t")
            self.term_dict[word.strip().lower()] = "CHESS_" + w_type.strip().lower()

    def term_match(self, term):
        """Find term match"""

        most_common = Counter([v for t, v in self.term_dict.iteritems() if t in term]).most_common(1)
        if most_common == []:
            most_common = None
        else:
            most_common = most_common[0][0]
        return most_common

    def count_terms(self, tokens):
        """Count number of matching terms"""

        terms = [self.term_match(t) for t in tokens ]
        
        terms = [t for t in terms if t != None]

        #print terms
        lf = dict(Counter(terms))
        for k in lf:
            lf[k] /= float(len(tokens))
            #lf[k] = 1 # binarize?
            pass
        return lf

def get_bin(bins, val):
    """Compute bin membership for a given value"""

    for i in xrange(len(bins)):
        if val < bins[i]:
            return i-1
    
    return i-1




def read_pgnx(data, verbose=False, min_annot=3):
    """Read pgnx file, filter games with fewer than min_annot annotations"""

    game_strs = data.split("<==>\n")
    games = []

    for game_str in game_strs:
        # try to parse the line
        try:
            player, rating, pgn_str = game_str.split("\n",2)
        except:
            if verbose:
                stderr.write("game unparsable: %s\n" % game_str)
            continue

        # remove punctuation from rating, e.g., 900+
        rating = rating.strip(punctuation)

        # skip non-numerical ratings
        if not rating.isdigit():
            if verbose:
                stderr.write("dropping game with non-integer rating: %s\n" % rating)
            continue
        rating = int(rating)

        # skip ratings that are unrealistic
        if rating > 3000 or rating < 500:
            if verbose:
                stderr.write("rating unrealistic: %d\n" % rating)
            continue

        # load the pgn and extract annotations
        game_pgn = pgn.loads(pgn_str)[0]
        annotations = [move for move in game_pgn.moves if move.strip().startswith("{")]
        if len(annotations) < min_annot:
            if verbose:
                stderr.write("dropping game with too few annotations\n")
            continue
        games.append((player, rating, game_pgn, annotations))
    
    return games


def write_results(gold, pred, ratings, text):
    """Write classifier results to file
    
    gold: list of true answers
    pred: list of predictions
    ratings: list of ratings
    text: list of annotations
    """

    f = open("results.txt", "w")
    for g, p, r, t in zip(gold, pred, ratings, text):
        f.write("%d\t%d\t%d\t%s\n" % (g,p,r,t))

    f.close()
