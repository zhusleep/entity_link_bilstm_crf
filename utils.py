import random, os, torch
import numpy as np
from sklearn.metrics import classification_report

def get_threshold(predict, label):
    thre_list = []
    for i in range(50):
        preds = sorted(list(predict[:, i].flatten()), reverse=True)
        n = sum(preds)
        m = 0
        e = 0
        f1 = 0
        cut_thre = 0
        for threshold in preds:
            e += threshold # 正例期望个数
            m += 1 #正例提交个数
            f1_temp = e/(m+n)
            if f1<f1_temp:
                f1=f1_temp
                cut_thre=threshold
        thre_list.append(cut_thre)
    print('阈值', thre_list)
    count = len(label)

    thre_list = np.array(thre_list).reshape(1, -1).repeat(count, axis=0)
    result = np.where(predict < thre_list, 0, 1)
    result_f = result.flatten()
    label_f = label.flatten()
    # # classification report
    # def cal_cli(m):
    #     temp = []
    #     for item in m:
    #         temp_row = []
    #         for i, row in enumerate(item):
    #             if row!=0:
    #                 temp_row.append(row)
    #             else:
    #                 temp_row.append(0)
    #
    # for j in range(result):

    # print(classification_report(label_f, result_f))

    hit = sum([((label_f[i] == result_f[i]) & (label_f[i] == 1)) for i in range(count*50)])
    acc = hit/sum(result_f)
    recall = hit/sum(label_f)
    f1 = 2*acc*recall/(acc+recall)
    INFO = 'acc %f, recall%f, f1% f' % (acc, recall, f1)
    return INFO


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def read_data(char=False):
    if not char:
        train_sentence = np.load('data/train_sentence.npy')
        label = np.load('data/train_label.npy')
        dev_sentence = np.load('data/dev_sentence.npy')
        dev_label = np.load('data/dev_label.npy')
        train_postag = np.load('data/train_postag.npy')
        dev_postag = np.load('data/dev_postag.npy')
        train_ner = np.load('data/train_ner.npy')
        dev_ner = np.load('data/dev_ner.npy')
    else:
        train_sentence = np.load('data/train_sentence_char.npy')
        label = np.load('data/train_label_char.npy')
        dev_sentence = np.load('data/dev_sentence_char.npy')
        dev_label = np.load('data/dev_label_char.npy')
        train_postag = np.load('data/train_postag_char.npy')
        dev_postag = np.load('data/dev_postag_char.npy')
        train_ner = np.load('data/train_ner_char.npy')
        dev_ner = np.load('data/dev_ner_char.npy')
    return train_sentence, train_postag, label, train_ner, dev_sentence, dev_postag, dev_label, dev_ner


def load_glove(embedding_file, max_features, tokenizer):

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.rstrip(' \n').split(" ")) for o in open(embedding_file) if len(o) > 100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    word_index = tokenizer.voc
    nb_words = min(max_features, len(word_index))
    print('词向量个数%d' % nb_words)
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    unknow = 0
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
        else:
            unknow += 1
    print('一共%d个oov'%unknow)
    return embedding_matrix


def sequence_cross_entropy_with_logits(logits: torch.FloatTensor,
                                       targets: torch.LongTensor,
                                       weights: torch.FloatTensor,
                                       average: str = "batch",
                                       label_smoothing: float = None) -> torch.FloatTensor:
    """
    Computes the cross entropy loss of a sequence, weighted with respect to
    some user provided weights. Note that the weighting here is not the same as
    in the :func:`torch.nn.CrossEntropyLoss()` criterion, which is weighting
    classes; here we are weighting the loss contribution from particular elements
    in the sequence. This allows loss computations for models which use padding.
    Parameters
    ----------
    logits : ``torch.FloatTensor``, required.
        A ``torch.FloatTensor`` of size (batch_size, sequence_length, num_classes)
        which contains the unnormalized probability for each class.
    targets : ``torch.LongTensor``, required.
        A ``torch.LongTensor`` of size (batch, sequence_length) which contains the
        index of the true class for each corresponding step.
    weights : ``torch.FloatTensor``, required.
        A ``torch.FloatTensor`` of size (batch, sequence_length)
    average: str, optional (default = "batch")
        If "batch", average the loss across the batches. If "token", average
        the loss across each item in the input. If ``None``, return a vector
        of losses per batch element.
    label_smoothing : ``float``, optional (default = None)
        Whether or not to apply label smoothing to the cross-entropy loss.
        For example, with a label smoothing value of 0.2, a 4 class classification
        target would look like ``[0.05, 0.05, 0.85, 0.05]`` if the 3rd class was
        the correct label.
    Returns
    -------
    A torch.FloatTensor representing the cross entropy loss.
    If ``average=="batch"`` or ``average=="token"``, the returned loss is a scalar.
    If ``average is None``, the returned loss is a vector of shape (batch_size,).
    """
    if average not in {None, "token", "batch"}:
        raise ValueError("Got average f{average}, expected one of "
                         "None, 'token', or 'batch'")

    # shape : (batch * sequence_length, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # shape : (batch * sequence_length, num_classes)
    log_probs_flat = torch.nn.functional.log_softmax(logits_flat, dim=-1)
    # shape : (batch * max_len, 1)
    targets_flat = targets.view(-1, 1).long()

    if label_smoothing is not None and label_smoothing > 0.0:
        num_classes = logits.size(-1)
        smoothing_value = label_smoothing / num_classes
        # Fill all the correct indices with 1 - smoothing value.
        one_hot_targets = torch.zeros_like(log_probs_flat).scatter_(-1, targets_flat, 1.0 - label_smoothing)
        smoothed_targets = one_hot_targets + smoothing_value
        negative_log_likelihood_flat = - log_probs_flat * smoothed_targets
        negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
    else:
        # Contribution to the negative log likelihood only comes from the exact indices
        # of the targets, as the target distributions are one-hot. Here we use torch.gather
        # to extract the indices of the num_classes dimension which contribute to the loss.
        # shape : (batch * sequence_length, 1)
        negative_log_likelihood_flat = - torch.gather(log_probs_flat, dim=1, index=targets_flat)
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood_flat.view(*targets.size())
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood * weights.float()

    if average == "batch":
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(1) / (weights.sum(1).float() + 1e-13)
        num_non_empty_sequences = ((weights.sum(1) > 0).float().sum() + 1e-13)
        return per_batch_loss.sum() / num_non_empty_sequences
    elif average == "token":
        return negative_log_likelihood.sum() / (weights.sum().float() + 1e-13)
    else:
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(1) / (weights.sum(1).float() + 1e-13)
        return per_batch_loss