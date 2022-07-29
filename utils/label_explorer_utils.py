""" Utility functions for label_explorer notebook. Written by Emily Nguyen, Brett Nelson, Nicholas Chin,
    Lawrence Berkeley national labs / U C Berkeley."""
def make_vectorized_labels(blist):
    """Vectorizes list of DLC video trial labels for use in ML-standard format
        Converts labels which hand and tug vs no tug string labels into numbers.

    Attributes
    -------------
    blist: list, of trial labels for a specific rat,date,session
            For more robust description please see github

    Returns
    ----------
    new_list: array, of lists of numeric labels.
            One list corresponds to one labeled trial.
    ind_total: array of lists of reaching indices .
            Currently all empty.

    """
    ll = len(blist)
    new_list = np.empty((ll, 9))
    ind_total = []
    for ix, l in enumerate(blist):
        if 'l' in str(l[5]):
            if 'lr' in str(l[5]):
                blist[ix][5] = 2
            else:
                blist[ix][5] = 1
        elif 'bi' in str(l[5]):
            if 'lbi' in str(l[5]):
                blist[ix][5] = 4
            else:
                blist[ix][5] = 3
        if 'r' in str(l[5]):
            blist[ix][5] = 0
        if l[5] == 0:
            blist[ix][5] = 5  # null trial
        try:
            if 'no' in str(l[6]):
                blist[ix][6] = 0
            else:
                blist[ix][6] = 1
        except:
            continue
        try:
            if len(l) > 9:  # are there indices?
                ind_total.append([l[9], l[10]])
            if len(l) > 11:  # second indices?
                ind_total.append([l[11], l[12]])
            if len(l) > 13:
                ind_total.append([l[13], l[14]])
            if len(l) > 15:
                ind_total.append([l[15], l[16]])
            if len(l) > 17:
                ind_total.append([l[17], l[18]])
            if len(l) > 19:
                ind_total.append([l[19], l[20]])
        except:
            print("index error", ix)
        new_list[ix, :] = blist[ix][0:9]
    return new_list, np.array(ind_total)

def make_vectorized_labels_to_df(labels):
    """Convert return value from make_vectorized_labels into a pandas df

    Args:
        labels (arr of lists): return value from make_vectorized_labels

    Returns:
        newdf(df)

    Examples:
        >>> l18l, ev18 = CU.make_vectorized_labels(l18)
        >>> make_vectorized_labels_to_df(l18l)
    """
    newdf = pd.DataFrame(data=labels,
                         columns=['Trial', 'Start Frame', 'Stop Frame', 'Trial Type',
                                  'Num Reaches', 'Which Hand', 'Tug', 'Hand Switch', 'Num Frames',
                                  'Date', 'Session', 'Rat'])
    return newdf


def onehot_nulls(type_labels_):
    # kwargs: n_f_fr_s_st: Trial type (null, failed, failed_rew,s ,succ_tug), label key [0, 1, 2, 3, 4]
    null_labels = np.zeros((type_labels_.shape[0]))
    null_labels[np.where(type_labels_ == 0)] = 1  # 1 if null, 0 if real trial
    return null_labels


def onehot_num_reaches(num_labels_):
    num_r_labels = np.zeros((num_labels_.shape[0]))  # 0 vector
    num_r_labels[np.where(num_labels_ > 1)] = 1  # 0 if <=1, 1 if > 1 reaches
    return num_r_labels


def hand_type_onehot(hand_labels_):
    hand_type_label = np.zeros((hand_labels_.shape[0]))
    hand_type_label[np.where(hand_labels_ > 1)] = 1  # classify all non r,l reaches as 1
    return hand_type_label


def multiple_reaches_onehot(labels):
    new_labels = np.zeros((labels.shape[0]))
    new_labels[np.where(labels >= 4)] = 2  # classify 4+, 2, 3 reaches
    new_labels[np.where(labels == 3)] = 1
    new_labels[np.where(labels == 2)] = 0
    return new_labels


# REVERSE
def inverse_transform_nulls(label):
    new_labels = np.repeat('not null', label.shape[0])
    new_labels[np.where(label == 1)] = 'null'
    return new_labels


def inverse_transform_reaches(label):
    new_labels = np.repeat('1 reach', label.shape[0])
    new_labels[np.where(label == 1)] = '2+'
    return new_labels


def inverse_transform_hand(label):
    new_labels = np.repeat('R, L', label.shape[0])
    new_labels[np.where(label == 1)] = 'LRA, RLA, Bi'
    return new_labels


def map_replace(df, col_name, fn):
    df = df.copy()
    col = df[col_name].values.astype(float)
    df[col_name] = fn(col)
    return df


def plt_to_pdf(figures, filename):
    # save multiple figures in one pdf file
    with matplotlib.backends.backend_pdf.PdfPages(filename) as pdf:
        for fig in figures:
            print(fig)
            pdf.savefig(fig)
