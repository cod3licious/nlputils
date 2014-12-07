import colorsys
import json
import numpy as np
import matplotlib.pyplot as plt
from nlputils.dict_utils import invert_dict0


def get_colors(N=100):
    HSV_tuples = [(x*1.0/N, 1., 0.8) for x in range(N)]
    return map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)

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
    return [np.nonzero(lscale>=val)[0][0] for val in X]

def pretty_coloring(X, varcol=0, N=100):
    """
    X: rows = observations, cols = variables
    varcol: variable for which different colors should be assigned
    N: number of colors that can be used
    """
    coloridx = colorindex(X[:,varcol], N)
    colors = np.array(get_colors(N))
    return colors[coloridx,:]

def prepare_viz(doc_ids, docdict, doccats, x, y, catdesc={}, filepath='docs.json'):
    """
    function to prepare text data for 2 dim visualization by saving a json file, that is a list of dicts,
    where each dict decodes 1 doc with "id" (doc_id), "x" and "y" (2dim coordinates derived from the kernel matrix
    using classical scaling), "title" (category/ies), "description" (whatever is in docdict at doc_id), "color" (for cat)
    Input:
        doc_ids: list with keys for docdict and doccats
        docdict: dict with docid:'description'
        doccats: dict with docid: cat
        x, y: 2d coordinates for all data points in the order of doc_ids (use x, y = proj2d(K, use_tsne, evcrit))
        catdesc: category descriptions
        filepath: where the json file will be saved
    """
    # pretty preprocessing
    categories = set(invert_dict0(doccats).keys())
    if not catdesc:
        catdesc = {cat:cat for cat in categories}    
    colorlist = get_colors(len(categories))
    colordict = {cat:(255*colorlist[i][0],255*colorlist[i][1],255*colorlist[i][2]) for i, cat in enumerate(sorted(categories))}
    # save as json
    print("saving json")
    data_json = []
    for i, key in enumerate(doc_ids):
        data_json.append({"id":key,"x":x[i],"y":y[i],"title":str(key)+" (%s)"%catdesc[doccats[key]],"description":docdict[key],"color":"rgb(%i,%i,%i)"%colordict[doccats[key]]})
    with open(filepath,"w") as f:
        f.write(json.dumps(data_json,indent=2))

def basic_viz(doc_ids, doccats, x, y, catdesc={}, title=''):
    """
    plot a scatter plot of the data in 2d
    Input:
        doc_ids: list with keys for docdict and doccats
        doccats: dict with docid: cat
        x, y: 2d coordinates for all data points in the order of doc_ids (use x, y = proj2d(K, use_tsne, evcrit))
        catdesc: category descriptions (for legend)
    """
    # pretty preprocessing
    categories = set(invert_dict0(doccats).keys())
    if not catdesc:
        catdesc = {cat:cat for cat in categories}    
    colorlist = get_colors(len(categories))
    colordict = {cat:(colorlist[i][0],colorlist[i][1],colorlist[i][2]) for i, cat in enumerate(sorted(categories))}
    # plot scatter plot
    plt.figure()
    for j, cat in enumerate(sorted(categories)):
        # get docids that belong to the current category
        didx_temp = [i for i, did in enumerate(doc_ids) if cat == doccats[did]]
        plt.plot(x[didx_temp], y[didx_temp], 'o', label=catdesc[cat], color=colordict[cat], alpha=0.5, markeredgewidth=0)
    plt.xticks([],[])
    plt.yticks([],[])
    plt.title(title)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), numpoints=1)
    #plt.tight_layout()
