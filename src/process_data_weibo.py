# encoding=utf-8
try:
    import cPickle as pickle
except ImportError:
    import pickle
import numpy as np
from torchvision import datasets, models, transforms
import os
from collections import defaultdict
import sys, re
import pandas as pd
from PIL import Image
import jieba
import os.path


def stopwordslist(filepath='../Data/weibo/stop_words.txt'):
    stopwords = {}
    for line in open(filepath, 'r').readlines():
        # line = unicode(line, "utf-8").strip()
        line = line.strip()
        stopwords[line] = 1
    # stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(u"[，。 :,.；|-“”——_/nbsp+&;@、《》～（）())#O！：【】]", "", string)
    return string.strip().lower()


def read_image():
    image_list = {}
    file_list = ['../Data/weibo/nonrumor_images/', '../Data/weibo/rumor_images/']
    for path in file_list:
        data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        for i, filename in enumerate(os.listdir(path)):  # assuming gif

            # print(filename)
            try:
                im = Image.open(path + filename).convert('RGB')
                im = data_transforms(im)
                # im = 1
                image_list[filename.split('/')[-1].split(".")[0].lower()] = im
            except:
                print(filename)
    print("image length " + str(len(image_list)))
    return image_list


def write_txt(data):
    f = open("../Data/weibo/top_n_data.txt", 'wb')
    for line in data:
        for l in line:
            f.write(l + "\n")
        f.write("\n")
        f.write("\n")
    f.close()


text_dict = {}


def write_data(flag, image, text_only):
    def read_post(flag):
        stop_words = stopwordslist()
        pre_path = "../Data/weibo/tweets/"
        file_list = [pre_path + "test_nonrumor.txt", pre_path + "test_rumor.txt", \
                     pre_path + "train_nonrumor.txt", pre_path + "train_rumor.txt"]
        if flag == "train":
            id = pickle.load(open("../Data/weibo/train_id.pickle", 'rb'))
        elif flag == "validate":
            id = pickle.load(open("../Data/weibo/validate_id.pickle", 'rb'))
        elif flag == "test":
            id = pickle.load(open("../Data/weibo/test_id.pickle", 'rb'))

        post_content = []
        labels = []
        image_ids = []
        twitter_ids = []
        data = []
        column = ['post_id', 'image_id', 'original_post', 'post_text', 'label', 'event_label']
        key = -1
        map_id = {}
        top_data = []
        for k, f in enumerate(file_list):

            f = open(f, 'r')
            if (k + 1) % 2 == 1:
                label = 0  # real is 0
            else:
                label = 1  # fake is 1

            twitter_id = 0
            line_data = []
            top_line_data = []

            for i, l in enumerate(f.readlines()):

                if (i + 1) % 3 == 1:
                    line_data = []
                    twitter_id = l.split('|')[0]
                    line_data.append(twitter_id)

                if (i + 1) % 3 == 2:
                    line_data.append(l.lower())

                if (i + 1) % 3 == 0:
                    l = clean_str_sst(l)
                    seg_list = jieba.cut_for_search(l)
                    new_seg_list = []
                    for word in seg_list:
                        if word not in stop_words:
                            new_seg_list.append(word)

                    clean_l = " ".join(new_seg_list)
                    if len(clean_l) > 10 and line_data[0] in id:  # truncrate small length
                        post_content.append(l)
                        line_data.append(l)
                        line_data.append(clean_l)
                        line_data.append(label)
                        event = int(id[line_data[0]])
                        if event not in map_id:
                            map_id[event] = len(map_id)
                            event = map_id[event]
                        else:
                            event = map_id[event]

                        line_data.append(event)

                        data.append(line_data)

            f.close()

        data_df = pd.DataFrame(np.array(data), columns=column)
        write_txt(top_data)

        return post_content, data_df

    post_content, post = read_post(flag)
    print("Original post length is " + str(len(post_content)))
    print("Original data frame is " + str(post.shape))

    def find_most(db):
        maxcount = max(len(v) for v in db.values())
        return [k for k, v in db.items() if len(v) == maxcount]

    def select(train, selec_indices):
        temp = []
        for i in range(len(train)):
            ele = list(train[i])
            temp.append([ele[i] for i in selec_indices])
            #   temp.append(np.array(train[i])[selec_indices])
        return temp

    def paired(text_only=False):
        ordered_image = []
        ordered_text = []
        ordered_post = []
        ordered_event = []
        label = []
        post_id = []
        image_id_list = []
        # image = []

        image_id = ""
        for i, id in enumerate(post['post_id']):
            for image_id in post.iloc[i]['image_id'].split('|'):
                image_id = image_id.split("/")[-1].split(".")[0]
                if image_id in image:
                    break

            if text_only or image_id in image:
                if not text_only:
                    image_name = image_id
                    image_id_list.append(image_name)
                    ordered_image.append(image[image_name])
                ordered_text.append(post.iloc[i]['original_post'])
                ordered_post.append(post.iloc[i]['post_text'])
                ordered_event.append(post.iloc[i]['event_label'])
                post_id.append(id)

                label.append(post.iloc[i]['label'])

        label = np.array(label, dtype=np.int)
        ordered_event = np.array(ordered_event, dtype=np.int)

        print("Label number is " + str(len(label)))
        print("Rummor number is " + str(sum(label)))
        print("Non rummor is " + str(len(label) - sum(label)))

        #
        if flag == "test":
            y = np.zeros(len(ordered_post))
        else:
            y = []

        data = {"post_text": np.array(ordered_post),
                "original_post": np.array(ordered_text),
                "image": ordered_image, "social_feature": [],
                "label": np.array(label), \
                "event_label": ordered_event, "post_id": np.array(post_id),
                "image_id": image_id_list}
        # print(data['image'][0])

        print("data size is " + str(len(data["post_text"])))

        return data

    paired_data = paired(text_only)

    print("paired post length is " + str(len(paired_data["post_text"])))
    print("paried data has " + str(len(paired_data)) + " dimension")
    return paired_data


def load_data(train, validate, test):
    vocab = defaultdict(float)
    all_text = list(train['post_text']) + list(validate['post_text']) + list(test['post_text'])
    for sentence in all_text:
        for word in sentence:
            vocab[word] += 1
    return vocab, all_text


def build_data_cv(data_folder, cv=10, clean_string=True):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    pos_file = data_folder[0]
    neg_file = data_folder[1]
    vocab = defaultdict(float)
    with open(pos_file, "rb") as f:
        for line in f:
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum = {"y": 1,
                     "text": orig_rev,
                     "num_words": len(orig_rev.split()),
                     "split": np.random.randint(0, cv)}
            revs.append(datum)
    with open(neg_file, "rb") as f:
        for line in f:
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum = {"y": 0,
                     "text": orig_rev,
                     "num_words": len(orig_rev.split()),
                     "split": np.random.randint(0, cv)}
            revs.append(datum)
    return revs, vocab


def get_W(word_vecs, k=32):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    # vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(len(word_vecs) + 1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs


def add_unknown_words(word_vecs, vocab, min_df=1, k=32):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)


def get_data(text_only):
    # text_only = False

    if text_only:
        print("Text only")
        image_list = []
    else:
        print("Text and image")
        image_list = read_image()

    train_data = write_data("train", image_list, text_only)
    validate_data = write_data("validate", image_list, text_only)
    test_data = write_data("test", image_list, text_only)

    return train_data, validate_data, test_data

if __name__ == '__main__':
    train, validate, test = get_data(text_only=False)
    print('finish!')


