from __future__ import unicode_literals, division, print_function, absolute_import
from builtins import range, str
import colorsys
import json
import numpy as np
import matplotlib.pyplot as plt
from .dict_utils import invert_dict0


def get_colors(N=100):
    HSV_tuples = [(x * 1. / (N+1), 1., 0.8) for x in range(N)]
    return [colorsys.hsv_to_rgb(*x) for x in HSV_tuples]


def colorindex(X, N=100):
    """
    X: list/vector for which different colors should be assigned
    N: number of colors that can be used
    """
    # get min & max values
    minv, maxv = min(X), max(X)
    # get a linear scale of the values
    lscale = np.linspace(minv, maxv, N)
    # for each value in X, find the index in the linspace
    return [np.nonzero(lscale >= val)[0][0] for val in X]


def pretty_coloring(X, varcol=0, N=100):
    """
    X: rows = observations, cols = variables
    varcol: variable for which different colors should be assigned
    N: number of colors that can be used
    """
    coloridx = colorindex(X[:, varcol], N)
    colors = np.array(get_colors(N))
    return colors[coloridx, :]


def prepare_viz(doc_ids, docdict, doccats, x, y, catdesc={}, filepath='docs.json', doc_ids_test=[], x_test=[], y_test=[]):
    """
    function to prepare text data for 2 dim visualization by saving a json file, that is a list of dicts,
    where each dict decodes 1 doc with "id" (doc_id), "x" and "y" (2dim coordinates derived from the kernel matrix
    using classical scaling), "title" (category/ies), "description" (whatever is in docdict at doc_id), "color" (for cat)
    Input:
        - doc_ids: list with keys for docdict and doccats
        - docdict: dict with docid:'description'
        - doccats: dict with docid: cat
        - x, y: 2d coordinates for all data points in the order of doc_ids (use x, y = proj2d(K, use_tsne, evcrit))
        - catdesc: category descriptions
        - filepath: where the json file will be saved
        - doc_ids_test, x_test, y_test: optional, same as before but for test points
    """
    # pretty preprocessing
    if not catdesc:
        categories = set(invert_dict0(doccats).keys())
        catdesc = {cat: cat for cat in categories}
    else:
        categories = list(catdesc.keys())
    colorlist = get_colors(len(categories))
    colordict = {cat: (255 * colorlist[i][0], 255 * colorlist[i][1], 255 * colorlist[i][2]) for i, cat in enumerate(sorted(categories))}
    # save as json
    print("saving json")
    data_json = []
    for i, key in enumerate(doc_ids):
        data_json.append({"id": key, "x": x[i], "y": y[i], "title": str(
            key) + " (%s)" % catdesc[doccats[key]], "description": docdict[key], "color": "rgb(%i,%i,%i)" % colordict[doccats[key]]})
    # if we have test points, do the same again
    for i, key in enumerate(doc_ids_test):
        data_json.append({"id": key, "x": x_test[i], "y": y_test[i], "title": str(key) + " (%s) - TEST POINT" % catdesc[
                         doccats[key]], "description": docdict[key], "color": "rgb(%i,%i,%i)" % colordict[doccats[key]]})
    with open(filepath, "w") as f:
        f.write(json.dumps(data_json, indent=2))


def basic_viz(doc_ids, doccats, x, y, catdesc={}, title='', doc_ids_test=[], x_test=[], y_test=[]):
    """
    plot a scatter plot of the data in 2d
    Input:
        - doc_ids: list with keys for docdict and doccats
        - doccats: dict with docid: cat
        - x, y: 2d coordinates for all data points in the order of doc_ids (use x, y = proj2d(K, use_tsne, evcrit))
        - catdesc: category descriptions (for legend)
        - doc_ids_test, x_test, y_test: optional, same as before but for test points (will get a higher alpha)
    """
    # pretty preprocessing
    if not catdesc:
        categories = set(invert_dict0(doccats).keys())
        catdesc = {cat: cat for cat in categories}
    else:
        categories = list(catdesc.keys())
    colorlist = get_colors(len(categories))
    colordict = {cat: (colorlist[i][0], colorlist[i][1], colorlist[i][2]) for i, cat in enumerate(sorted(categories))}
    # plot scatter plot
    plt.figure()
    for j, cat in enumerate(sorted(categories)):
        # get docids that belong to the current category
        didx_temp = [i for i, did in enumerate(doc_ids) if cat == doccats[did]]
        plt.plot(x[didx_temp], y[didx_temp], 'o', label=catdesc[cat],
                 color=colordict[cat], alpha=0.6, markeredgewidth=0)
        # possibly do the same for test points
        if doc_ids_test:
            didx_temp = [i for i, did in enumerate(doc_ids_test) if cat == doccats[did]]
            plt.plot(x_test[didx_temp], y_test[didx_temp], 'o', label=catdesc[
                     cat], color=colordict[cat], alpha=1., markeredgewidth=0)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.xlim([x.min(), x.max()])
    plt.ylim([y.min(), y.max()])
    # plt.axis('equal')
    plt.title(title, fontsize=16)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), numpoints=1)
    # plt.tight_layout()


def json2plot(jsonpath, title='', baseline=False):
    data_json = json.load(open(jsonpath))
    doc_ids = [d["id"] for d in data_json]
    x = np.array([d["x"] for d in data_json])
    y = np.array([d["y"] for d in data_json])
    if baseline:
        catdesc = {"moviereviews neg": 1,
                   "moviereviews pos": 2,
                   "ohsumed C04": 3,
                   "ohsumed C10": 4,
                   "ohsumed C14": 5,
                   "ohsumed C23": 6,
                   "reuters coffee": 7,
                   "reuters crude": 8,
                   "reuters gold": 9,
                   "reuters grain": 10,
                   "reuters interest": 11,
                   "reuters money-supply": 12,
                   "reuters sugar": 13,
                   "reuters trade": 14,
                   "pmcpar abs-int-dis": 15,
                   "pmcpar methods": 16,
                   "pmcpar results": 17}
        doccats = {d["id"]: catdesc[d["title"].split("(")[-1][:-1]] for d in data_json}
        basic_viz(doc_ids, doccats, x, y, {v: k for k, v in catdesc.items()}, title)
    else:
        doccats = {d["id"]: d["title"].split("(")[-1][:-1] for d in data_json}
        basic_viz(doc_ids, doccats, x, y, {}, title)
