import csv
import regex
from nltk import download as nltk_download
from math import log
nltk_download('punkt')
nltk_download('stopwords')
nltk_download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


categories = {
                'TRAVEL': 0.74,
                'BUSINESS': 0.84,
                'STYLE & BEAUTY': 0.77,
}

INDEX ='index'
AUTHOR ='authors'
CAT = 'category'
DATE = 'date'
HL = 'headline'
LINK = 'link'
DESC = 'short_description'


def get_data():
    global categories,HL, CAT, DESC
    data_per_cat = {cat: [] for cat in categories}
    with open('../Attachment/data.csv') as f:
        reader = csv.DictReader(f)
        for line in reader:
            h = line[HL] + ' '
            cat = line[CAT]
            if cat in categories:
                data_per_cat[cat].append(h+line[DESC])

    train_data = {}
    evaluation_data = {}
    for cat in categories:
        length = len(data_per_cat[cat])
        precision = categories[cat]
        train_data[cat] = data_per_cat[cat][:int(length*precision)]
        evaluation_data[cat] = data_per_cat[cat][int(length*precision):]
    return train_data, evaluation_data


def get_clean_words(text):
    stop_words = set(stopwords.words('english'))
    lm = WordNetLemmatizer()
    words = word_tokenize(text)
    words = list(map(lambda w: w.lower(), words))
    words = list(filter(lambda w: w not in stop_words, words))
    words = list(filter(lambda w: regex.fullmatch('[a-zA-Z]+', w), words))
    words = list(map(lambda w: lm.lemmatize(w), words))
    return words


def train(train_data):
    global categories
    cat_prob = {'prob': {}}
    total_words = 0
    for cat in categories:
        cat_prob[cat] = {}
        word_num = 0
        for desc in train_data[cat]:
            clean_words = get_clean_words(desc)
            for word in clean_words:
                word_num += 1
                if word in cat_prob[cat]:
                    cat_prob[cat][word] += 1
                else:
                    cat_prob[cat][word] = 1
        total_words += word_num
        log_words_count = log(word_num)
        for word in cat_prob[cat]:
            log_word_num = log(cat_prob[cat][word])
            cat_prob[cat][word] = log_word_num - log_words_count
        cat_prob['prob'][cat] = word_num

    for cat in categories:
        cat_prob['prob'][cat] = log(cat_prob['prob'][cat]) - log(total_words)

    return cat_prob


def guess_category(sentence, cat_prob):
    global categories
    words = get_clean_words(sentence)
    max_prob = float('-Inf')
    guessed_category = ''
    for cat in categories:
        sum = 0
        for word in words:
            if word in cat_prob[cat]:
                sum += cat_prob[cat][word]
            else:
                sum += -100
        sum += cat_prob['prob'][cat]
        if sum > max_prob:
            max_prob = sum
            guessed_category = cat
    return guessed_category


def evaluate(evaluation_data, cat_prob):
    global categories
    detected_cat = {cat: {"correct": 0, "all": 0, 'TRAVEL': 0,  'BUSINESS': 0, 'STYLE & BEAUTY': 0} for cat in categories}
    correct_cat = {cat: {"correct": 0, "all": 0} for cat in categories}
    for cat in categories:
        for desc in evaluation_data[cat]:
            detect = guess_category(desc, cat_prob)
            correct = detect == cat
            if correct:
                detected_cat[detect]["correct"] += 1
                correct_cat[cat]["correct"] += 1

            detected_cat[detect]["all"] += 1
            detected_cat[detect][cat] += 1
            correct_cat[cat]["all"] += 1
    for cat in categories:
        print('{} recall: {}%'.format(cat, int(correct_cat[cat]['correct']*100/correct_cat[cat]['all'])))
        print('{} precision: {}%'.format(cat, int(detected_cat[cat]['correct']*100/detected_cat[cat]['all'])))

    all = 0
    correct = 0
    for cat in categories:
        all += detected_cat[cat]['all']
        correct += detected_cat[cat]['correct']

    print('accuracy: {}%'.format(int(correct*100/all)))
    print(detected_cat)


def test(cat_prob):
    global categories, INDEX, HL, DESC, CAT
    with open('../Attachment/test.csv') as f, open('../output.csv', 'w') as o:
        writer = csv.DictWriter(o, fieldnames=[INDEX, CAT])
        writer.writeheader()
        reader = csv.DictReader(f)
        for line in reader:
            if line[DESC]:
                h = line[HL] + ' '
                detected_category = guess_category(h+line[DESC], cat_prob)
                writer.writerow({INDEX:line[INDEX], CAT:detected_category})


if __name__ == '__main__':
    train_data, evaluation_data = get_data()
    cat_prob = train(train_data)
    evaluate(evaluation_data, cat_prob)
    test(cat_prob)
